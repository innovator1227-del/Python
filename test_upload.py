import http.client
import uuid
from pathlib import Path

file_path = Path('test_upload_sample.txt')
file_path.write_text('Hello world upload test', encoding='utf-8')

with open(file_path, 'rb') as f:
    file_data = f.read()

boundary = '----WebKitFormBoundary' + uuid.uuid4().hex
body = []
body.append(f'--{boundary}')
body.append('Content-Disposition: form-data; name="file"; filename="test_upload_sample.txt"')
body.append('Content-Type: text/plain\r\n')
body.append(file_data.decode('utf-8'))
body.append(f'--{boundary}--\r\n')
body_bytes = '\r\n'.join(body).encode('utf-8')

conn = http.client.HTTPConnection('127.0.0.1', 5000)
conn.request(
    'POST',
    '/api/upload',
    body_bytes,
    {
        'Content-Type': f'multipart/form-data; boundary={boundary}',
        'Content-Length': str(len(body_bytes))
    }
)
resp = conn.getresponse()
print('STATUS', resp.status)
print(resp.read().decode('utf-8', errors='replace'))
