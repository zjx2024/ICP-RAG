import copy
import unittest

from backend.agent.citation_binding import CitationBindingError, citation_spans, inspect_record_citations, validate_record_citations


def calculation(code, value):
    return {'doc_id': 'source', 'record_id': code, 'doc_type': 'rbd',
            'authority': 'deterministic_derivation', 'review_status': 'auto_validated',
            'metadata': {'numeric_complete': True, 'calculation': {
                'parameters': {'mission_time': 12, 'mission_time_unit': 'h'},
                'results': {'system_reliability': value, 'blocks': {'unrelated': .1234}}}}}


class AuthorOwnedCitationsTests(unittest.TestCase):
    def setUp(self):
        self.ledger = {7: calculation('MODEL-AX', .91827), 13: calculation('MODEL-BY', .81234)}

    def check(self, text, **kwargs):
        return validate_record_citations(text, kwargs.get('ledger', self.ledger),
                                        calculation_requests=kwargs.get('requests', ['rbd_reliability']))

    def test_correct_same_paragraph_comparison_preserves_text_and_offsets(self):
        text = 'MODEL-AX 为0.91827[^7]，MODEL-BY 为0.81234[^13]；后者较低。'
        result = self.check(text)
        self.assertEqual([[7], [13], []], [s['citation_ids'] for s in result])
        self.assertTrue(all(s['text'] == text[s['start']:s['end']] for s in result))

    def test_swapped_values_or_ids_cannot_borrow_neighbour_even_in_same_paragraph(self):
        for text in ('MODEL-AX 为0.81234[^7]，MODEL-BY 为0.91827[^13]。',
                     'MODEL-AX 为0.91827[^13]，MODEL-BY 为0.81234[^7]。',
                     'MODEL-AX 为0.91827，MODEL-BY 为0.81234[^7][^13]。'):
            with self.subTest(text=text), self.assertRaises(CitationBindingError):
                self.check(text)

    def test_shared_parameter_is_not_ambiguous_model_result(self):
        self.check('任务时间均为12 h[^7][^13]。')
        with self.assertRaises(CitationBindingError):
            self.check('任务时间均为13 h[^7][^13]。')

    def test_unknown_and_unapproved_citations_stay_errors(self):
        with self.assertRaises(CitationBindingError):
            self.check('结果0.91827[^99]')
        with self.assertRaises(CitationBindingError):
            self.check('结果0.91827[^7]', ledger={})

    def test_binding_feedback_locates_identity_without_choosing_replacement(self):
        text = 'MODEL-BY 的结果为0.81234[^7]。'
        with self.assertRaises(CitationBindingError) as caught:
            self.check(text)
        error = caught.exception
        self.assertEqual('MODEL-BY 的结果为0.81234[^7]', error.span['text'])
        self.assertEqual([7], error.span['citation_ids'])
        self.assertEqual(['MODEL-BY'], error.unsupported_identities)
        self.assertNotIn('replacement', error.__dict__)

    def test_numeric_and_authority_feedback_also_locates_failing_span(self):
        for text, requests in (('结果0.81234[^7]', ['rbd_reliability']),
                               ('结果0.91827[^7]', [])):
            with self.subTest(text=text, requests=requests), self.assertRaises(CitationBindingError) as caught:
                self.check(text, requests=requests)
            self.assertEqual(text, caught.exception.span['text'])
            self.assertEqual([7], caught.exception.span['citation_ids'])

    def test_intermediate_value_cannot_masquerade_as_system_result(self):
        with self.assertRaises(CitationBindingError):
            self.check('结果0.1234[^7]')

    def test_computation_authority_cannot_be_removed_by_changing_wording(self):
        for text in ('Result: 0.91827[^7]', '得到0.91827[^7]', '0.91827[^7]'):
            with self.subTest(text=text), self.assertRaises(CitationBindingError):
                self.check(text, requests=[])

    def test_provisional_incomplete_and_model_only_derivations_cannot_publish(self):
        for key, value in (('validation_status', 'provisional'), ('numeric_complete', False),
                           ('usage_scope', 'model_content_only')):
            ledger = copy.deepcopy(self.ledger)
            ledger[7]['metadata'][key] = value
            with self.subTest(key=key), self.assertRaises(CitationBindingError):
                self.check('结果0.91827[^7]', ledger=ledger)

    def test_markdown_is_not_business_identifier_content(self):
        self.check('**MODEL-AX** 为 **0.91827**[^7]，`MODEL-BY` 为0.81234[^13]。')
        self.assertEqual([], citation_spans(''))

    def test_fta_cut_set_count_does_not_borrow_probability_permission(self):
        item = calculation('TREE-79', .91)
        item['doc_type'] = 'fta'
        item['metadata']['numeric_complete'] = False
        item['metadata']['calculation']['results'] = {'minimal_cut_set_count': 2, 'top_event_probability': .19}
        self.check('最小割集共2个[^7]', ledger={7: item}, requests=['fta_cut_sets'])
        with self.assertRaises(CitationBindingError):
            self.check('顶事件概率0.19[^7]', ledger={7: item}, requests=['fta_cut_sets'])

    def test_cut_set_table_row_index_is_not_an_engineering_value(self):
        # FTA-02: a cut-set table repeats "| 1 |", "| 2 |" ... as row numbers.
        # Only a consecutive 1..N index column is presentation; a lone leading
        # integer stays a real value and is still checked.
        item = calculation('TREE-79', .91)
        item['doc_type'] = 'fta'
        item['metadata']['numeric_complete'] = False
        item['metadata']['calculation']['results'] = {'minimal_cut_sets': [['A路间歇停机'], ['风扇堵转']],
                                                      'minimal_cut_set_count': 2}
        table = ('| 序号 | 最小割集 | 门结构 |\n| --- | --- | --- |\n'
                 '| 1 | **A路间歇停机** | 顶事件 OR 门直接输入 [^7]\n'
                 '| 2 | **风扇堵转** | 顶事件 OR 门直接输入 [^7]')
        self.check(table, ledger={7: item}, requests=['fta_cut_sets'])
        # A real mismatch inside the table is still caught.
        with self.assertRaises(CitationBindingError):
            self.check(table.replace('顶事件 OR 门直接输入 [^7]', '概率 0.77 [^7]', 1),
                       ledger={7: item}, requests=['fta_cut_sets'])
        # A single table row's leading integer is not a consecutive index.
        with self.assertRaises(CitationBindingError):
            self.check('| 3 | 概率 0.77 | 结果 [^7]', ledger={7: item}, requests=['fta_cut_sets'])
        # Non-table prose keeps every literal.
        with self.assertRaises(CitationBindingError):
            self.check('第 3 个割集为 0.77[^7]', ledger={7: item}, requests=['fta_cut_sets'])

    def test_cut_set_list_derived_counts_are_supported(self):
        # FTA-02: "8 of the 9 cut sets are single-event" counts the cited
        # credential's own list; that arithmetic needs no stored result field.
        item = calculation('TREE-79', .91)
        item['doc_type'] = 'fta'
        item['metadata']['numeric_complete'] = False
        item['metadata']['calculation']['results'] = {
            'minimal_cut_sets': [['A路间歇停机'], ['B路间歇停机'], ['风扇堵转'],
                                 ['温控闭环失效', '转速反馈丢失']],
            'minimal_cut_set_count': 4}
        self.check('最小割集共 4 个，其中 3 个为单事件割集[^7]',
                   ledger={7: item}, requests=['fta_cut_sets'])
        # A count that matches neither the stored nor any derived count fails.
        with self.assertRaises(CitationBindingError):
            self.check('最小割集共 4 个，其中 2 个为单事件割集[^7]',
                       ledger={7: item}, requests=['fta_cut_sets'])
        # Derived counts do not leak into probability authority.
        with self.assertRaises(CitationBindingError):
            self.check('概率 3[^7]', ledger={7: item}, requests=['fta_probability'])

    def test_distinct_revisions_do_not_share_result_values(self):
        ledger = copy.deepcopy(self.ledger)
        ledger[13]['record_id'] = ledger[7]['record_id']
        ledger[7]['metadata']['revision_id'] = 'REV-A7'
        ledger[13]['metadata']['revision_id'] = 'REV-B8'
        self.check('REV-A7结果0.91827[^7]，REV-B8结果0.81234[^13]', ledger=ledger)
        for text in ('REV-A7结果0.81234[^7]', 'REV-B8结果0.91827[^7]'):
            with self.assertRaises(CitationBindingError):
                self.check(text, ledger=ledger)

    def test_all_independent_binding_errors_are_collected_without_repair(self):
        text = 'MODEL-BY 为0.77[^7]。\n结果0.22[^13]。\n结果未知[^999]。'
        spans, errors = inspect_record_citations(text, self.ledger, calculation_requests=[])
        self.assertEqual(3, len({error.span['start'] for error in errors}))
        messages = [str(error) for error in errors]
        self.assertIn('cited span contains identity not supported by its records', messages)
        self.assertEqual(2, messages.count('calculation not authorized'))
        self.assertEqual(2, messages.count('numeric value not supported by this calculation'))
        self.assertIn('unknown or unapproved citation', messages)
        self.assertTrue(all(span['text'] == text[span['start']:span['end']] for span in spans))

    def test_joint_record_citations_keep_both_sources_and_do_not_share_globally(self):
        from backend.agent.tools.icp_diagnosis import _direct_evidence
        rows = [
            {'doc_id': 'manual', 'content_type': 'function_component', 'component_id': 'MOD-019',
             'name': '同名模块', 'description': '部件职责'},
            {'doc_id': 'manual', 'function_id': 'FUNC-8X', 'module_id': 'MOD-019',
             'module_name': '同名模块', 'function_name': '数据保存', 'failure_symptoms': '读取失败'},
            {'doc_id': 'manual', 'function_id': 'FUNC-9X', 'module_id': 'MOD-020',
             'module_name': '同名模块', 'function_name': '数据保存', 'failure_symptoms': '写入失败'},
        ]
        ledger = dict(zip((3, 7, 11), _direct_evidence('function_locator', rows)))
        text = 'MOD-019 对应FUNC-8X，可能读取失败。[^3][^7]'
        spans = validate_record_citations(text, ledger)
        self.assertEqual([3, 7], spans[0]['citation_ids'])
        self.assertEqual(text, spans[0]['text'])
        # An ID elsewhere in the answer, or an identically named neighbour,
        # cannot supply a missing source binding for this sentence.
        for bad in ('MOD-019 可能写入失败。[^11]',
                    'MOD-019。[^3]\nMOD-019 可能写入失败。[^11]'):
            with self.assertRaises(CitationBindingError):
                validate_record_citations(bad, ledger)
