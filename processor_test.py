from core.preprocessor import DocumentProcessor
from pathlib import Path

path = Path('test_upload_sample.txt')
path.write_text('Hello world upload test', encoding='utf-8')

processor = DocumentProcessor()
print('Processor initialized')
try:
    raw_text, tokens = processor.process_document(str(path))
    print('RAW_TEXT:', raw_text)
    print('TOKENS:', tokens)
    print('TOKEN COUNT:', len(tokens))
except Exception as exc:
    print('EXCEPTION TYPE:', type(exc).__name__)
    print('EXCEPTION:', exc)
    import traceback
    traceback.print_exc()
