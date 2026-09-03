import unittest

from backend.core.parsers.records import SourceRef
from backend.core.parsers.table_engine import (
    TableSchema,
    TableSource,
    extract_records,
    find_headers,
    normalize_header,
)


class TableEngineTests(unittest.TestCase):
    def test_header_normalization_uses_unicode_nfkc(self):
        self.assertEqual(normalize_header("ＨＭＣ　代码"), normalize_header("HMC代码"))

    def test_finds_header_after_long_cover_prefers_explicit_code_and_joins_continuation(self):
        rows = [[f"封面字段{index}", f"封面值{index}", ""] for index in range(12)]
        rows.extend([
            ["序号", "代码", "故障描述及现象"],
            ["1", "FM-001", "第一段"],
            ["", "", "续行说明"],
            ["序号", "代码", "故障描述及现象"],
            ["2", "FM-002", "第二条"],
        ])
        schema = TableSchema.from_mapping(
            "sample",
            {
                # Both columns can map to code; alias priority must make 代码 win.
                "code": ["代码", "序号"],
                "description": ["故障描述", "故障描述及现象"],
            },
            required_fields=("description",),
            identity_groups=(("code",),),
            min_mapped_fields=2,
            continuation_fields=("description",),
        )
        source = TableSource(
            rows=rows,
            source=SourceRef(file_name="sample.xlsx", element_type="table", sheet_name="Sheet1", table_index=1),
            row_numbers=list(range(1, len(rows) + 1)),
        )

        headers = find_headers(rows, schema)
        records = extract_records(source, schema)

        self.assertEqual(len(headers), 2)
        self.assertEqual([record.values["code"] for record in records], ["FM-001", "FM-002"])
        self.assertEqual(records[0].values["description"], "第一段\n续行说明")
        self.assertEqual(len(records[0].source_refs), 2)
        self.assertIn("row=14", str(records[0].source_refs[0]))

    def test_recognizes_two_level_repeated_hmc_header(self):
        rows = [
            ["HMC 代码", "HMC 代码", "HMC 代码", "故障描述现象", "故障分析程序/维修方法"],
            ["LCN", "HAMC", "HSC", "故障描述现象", "故障分析程序/维修方法"],
            ["00123456", "**", "0001", "显示异常", "检查显示模块"],
        ]
        schema = TableSchema.from_mapping(
            "hmc",
            {
                "hmc_code": ["HMC代码", "HMC"],
                "lcn": ["LCN"],
                "hamc": ["HAMC"],
                "hsc": ["HSC"],
                "fault_desc": ["故障描述现象", "故障现象"],
                "repair_method": ["故障分析程序/维修方法", "维修方法"],
            },
            required_fields=("fault_desc",),
            identity_groups=(("hmc_code",), ("lcn", "hsc")),
            min_mapped_fields=3,
        )
        source = TableSource(
            rows=rows,
            source=SourceRef(file_name="hmc.docx", element_type="table", table_index=1),
            row_numbers=[1, 2, 3],
        )

        records = extract_records(source, schema)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].header.end, 1)
        self.assertEqual(records[0].values["lcn"], "00123456")
        self.assertEqual(records[0].values["hsc"], "0001")

    def test_keeps_field_level_raw_formula_merge_and_source_metadata(self):
        from openpyxl import Workbook

        workbook = Workbook()
        sheet = workbook.active
        sheet.append(["代码", "故障描述"])
        sheet.append(["A-001", "=\"显示\"&\"异常\""])
        schema = TableSchema.from_mapping(
            "metadata",
            {"code": ["代码"], "description": ["故障描述"]},
            required_fields=("description",),
            identity_groups=(("code",),),
        )
        import tempfile
        from pathlib import Path
        from backend.core.parsers.table_engine import table_sources

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "formula.xlsx"
            workbook.save(path)
            record = extract_records(table_sources(str(path))[0], schema)[0]

        metadata = record.raw_data["field_cell_metadata"]["description"]
        self.assertEqual(metadata["formula"], '=\"显示\"&\"异常\"')
        self.assertEqual(metadata["raw_value"], '=\"显示\"&\"异常\"')
        self.assertIn("row=2", metadata["source_ref"])
        self.assertFalse(metadata["is_merge_continuation"])


if __name__ == "__main__":
    unittest.main()
