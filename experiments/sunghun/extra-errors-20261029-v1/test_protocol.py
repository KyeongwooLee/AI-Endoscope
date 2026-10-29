import hashlib
import tempfile
import unittest
from pathlib import Path

from protocol import commit,verify,read_records

class Checkpoints(unittest.TestCase):
    def test_backed_up_chunk_detects_corruption(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root=Path(directory)
            receipt=commit(root,'generations_000000',{'records.jsonl':b'{"id":"a"}\n'},
                           {'kind':'generations','count':1,'ids':['a'],'selected_errors':1})
            self.assertEqual(list(read_records(root)),[{'id':'a'}])
            path=root/'outbox/generations_000000.zip'
            with path.open('r+b') as stream:
                stream.seek(10)
                value=stream.read(1)
                stream.seek(10)
                stream.write(bytes([value[0]^1]))
            with self.assertRaises(AssertionError): verify(path,receipt)

    def test_immutable_checkpoint_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root=Path(directory)
            commit(root,'generations_000000',{'records.jsonl':b'{"id":"a"}\n'},
                   {'kind':'generations','count':1,'ids':['a'],'selected_errors':0})
            with self.assertRaises(FileExistsError):
                commit(root,'generations_000000',{'records.jsonl':b'{}\n'}, {})

if __name__=='__main__': unittest.main()
