import unittest,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from scoring import extract,recalculate
from learn import inputs,availability,predict_policies,best_thresholds
class Safeguards(unittest.TestCase):
 def test_scoring_span_and_refusal(self):
  for text,value in [('Mike will have $800 after spending on a shirt that costs $200.','8E+2'),('Final answer: Elizabeth can buy 5 pencils with 20 dollars if she wants 6 pens.','5'),('Final answer: insufficient information',None),('Final answer: 12.5','12.5')]:
   p=extract(text);self.assertEqual(p['answer'],value)
   if p['span']:self.assertTrue(any(c.isdigit() for c in text[slice(*p['span'])]))
  self.assertIsNone(extract('Calculation 2 + 2 = 4',True)['answer'])
 def test_prose_and_next_line_regressions(self):
  for text,value in [('Therefore, the final answer is:\n\nJosh had $9 at first.','9'),('Mary has 50 apples. Therefore, the final answer is 30.','3E+1'),('Final answer: The 2 boxes contain a total of 2304 soaps.','2304'),('Final answer: -$15','-15')]:
   self.assertEqual(extract(text)['answer'],value)
 def test_mask_and_conservative(self):
  p=np.array([[.6,.99,.65,.7],[.8,.99,.1,.1]]);m=np.array([[True,False,True,True],[True,False,True,True]]);settings={'margin':.2,'trigger_action':1,'trigger_threshold':.5}
  choices=predict_policies(np.array([.9,.9]),p,m,settings)
  np.testing.assert_array_equal(choices['selector'],[3,0]);np.testing.assert_array_equal(choices['conservative'],[0,0]);np.testing.assert_array_equal(choices['trigger'],[0,0])
 def test_calculator_unmodified(self):
  self.assertFalse(recalculate('x + 3 = 7\nFinal answer: 7')['available'])
  self.assertEqual(recalculate('2 * 3 = 7\nFinal answer: 7')['answer'],'6')
 def test_inputs_invariant_to_outcomes(self):
  import json,copy
  root=Path(__file__).resolve().parents[2];files=list((root/'runs/20261012_followup/records').glob('*.json'))
  if not files:self.skipTest('No pilot records yet')
  r=json.loads(files[0].read_text());other=copy.deepcopy(r);other.update(gold='99999',label_error=9,actions={'poison':'not features'})
  self.assertEqual(inputs([r]),inputs([other]))
if __name__=='__main__':unittest.main()
