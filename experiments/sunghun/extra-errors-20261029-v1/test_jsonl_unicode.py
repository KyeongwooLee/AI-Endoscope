import json
import unittest
from audit_local import loads_jsonl

class JsonlUnicode(unittest.TestCase):
    def test_unicode_line_separators_inside_json_strings(self):
        rows=[{'question':'A\u2028B\u0085C\u2029D','response':'x\ny'}, {'question':'next'}]
        encoded=('\n'.join(json.dumps(row,ensure_ascii=False) for row in rows)+'\n').encode('utf-8')
        self.assertEqual(loads_jsonl(encoded),rows)

if __name__=='__main__': unittest.main()
