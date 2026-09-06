import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import ts from 'typescript';

// The module has only type imports. Compile the actual production source with
// the installed TypeScript compiler; the production build separately typechecks it.
const source = await readFile(new URL('../src/api/diagnosisStream.ts', import.meta.url), 'utf8');
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ES2022 },
});
const { parseDiagnosisEvent, parseSseChunk, readDiagnosisStream } = await import(
  `data:text/javascript;base64,${Buffer.from(outputText).toString('base64')}`
);

function stream(chunks) {
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(chunk);
      controller.close();
    },
  });
}

async function collect(body) {
  const events = [];
  for await (const event of readDiagnosisStream(body)) events.push(event);
  return events;
}

test('LF, CRLF, comments and multiline data preserve typed events', () => {
  const { events, rest } = parseSseChunk(': heartbeat\r\n\r\nevent: answer\r\ndata: {"type":"answer",\r\ndata: "content":"证据"}\r\n\r\ndata: {"type":"done"}\n\npartial');
  assert.deepEqual(events.map((event) => event.type), ['answer', 'done']);
  assert.equal(events[0].content, '证据');
  assert.equal(rest, 'partial');
});

test('every byte boundary, including UTF-8 and CRLF splits, is safe', async () => {
  const bytes = new TextEncoder().encode('data: {"type":"answer_delta","content":"多轮问答"}\r\n\r\ndata: {"type":"done"}\r\n\r\n');
  const events = await collect(stream(Array.from(bytes, (byte) => Uint8Array.of(byte))));
  assert.equal(events[0].content, '多轮问答');
  assert.equal(events.at(-1).type, 'done');
});

test('partial answer is a normal completion state, not a server error', () => {
  const event = parseDiagnosisEvent({ type: 'answer', content: '来源仍需核对', completion_status: 'partial', unconfirmed_fields: ['api_name'], citations: [], used_evidence_ids: [] });
  assert.equal(event.completion_status, 'partial');
  assert.deepEqual(event.unconfirmed_fields, ['api_name']);
});

test('wrong evidence, status and text types cannot reach rendering as any', () => {
  for (const event of [
    { type: 'answer', content: {} },
    { type: 'citations', items: ['forged'] },
    { type: 'answer', content: 'x', citations: [{ title: {} }] },
    { type: 'answer', content: 'x', completion_status: 'invented' },
    { type: 'done', used_evidence_ids: [4] },
    { type: 'tool_call', params: [] },
    { type: 'stage', count: Number.NaN },
  ]) assert.throws(() => parseDiagnosisEvent(event), /格式异常/);
});

test('catalog rows without a public citation number remain valid', () => {
  const event = parseDiagnosisEvent({ type: 'done', evidence_catalog: [{ evidence_id: 'e-7', answer_citation_id: null, title: '候选' }] });
  assert.equal(event.evidence_catalog[0].answer_citation_id, null);
});

test('unknown extension event is ignored, malformed JSON is reported', () => {
  assert.equal(parseDiagnosisEvent({ type: 'future_extension' }), null);
  assert.throws(() => parseSseChunk('data: {broken}\n\n'), /JSON/);
});

test('EOF without done is an interrupted stream, even after answer text', async () => {
  const body = stream([new TextEncoder().encode('data: {"type":"answer","content":"部分文字"}\n\n')]);
  await assert.rejects(collect(body), /未收到完成确认/);
  assert.equal(body.locked, false);
});

test('an explicit stopped event is terminal and releases the reader', async () => {
  const body = stream([new TextEncoder().encode('data: {"type":"stopped"}\n\n')]);
  const events = await collect(body);
  assert.equal(events[0].type, 'stopped');
  assert.equal(body.locked, false);
});

test('reader failure is preserved and cleanup does not mask it', async () => {
  const body = new ReadableStream({ start(controller) { controller.error(new Error('network reset')); } });
  await assert.rejects(collect(body), /network reset/);
  assert.equal(body.locked, false);
});

test('consumer rejection cancels the reader and does not swallow the error event', async () => {
  let cancelled = false;
  const body = new ReadableStream({
    start(controller) { controller.enqueue(new TextEncoder().encode('data: {"type":"error","message":"来源不可用"}\n\n')); },
    cancel() { cancelled = true; },
  });
  await assert.rejects(async () => {
    for await (const event of readDiagnosisStream(body)) {
      if (event.type === 'error') throw new Error(event.message);
    }
  }, /来源不可用/);
  assert.equal(cancelled, true);
  assert.equal(body.locked, false);
});
