import unittest
from common import answer,calculate,recalculate,verdict,correct,strict_answer
class Invariants(unittest.TestCase):
    def test_scoring(self):
        self.assertTrue(correct(answer('Final answer: $1,200')[0],'1200'))
        self.assertEqual(answer('Therefore, 40 dollars.')[0],'4E+1')
        self.assertEqual(answer(r'Answer: \\boxed{12}')[0],'12')
        self.assertIsNone(answer('No numeric answer')[0])
        self.assertIsNone(answer('Intermediate 42',truncated=True)[0])
        self.assertEqual(answer('Final answer: 42',truncated=True)[0],'42')
        self.assertIsNone(strict_answer('Maybe 40'))
        self.assertEqual(answer('Therefore Sally will need 5844 square inches for 8 napkins.')[0],'5844')
        self.assertEqual(answer('Therefore 8 napkins cost 40 dollars.')[0],'4E+1')
    def test_verdict_exact(self):
        self.assertEqual(verdict('INCORRECT'),'INCORRECT')
        self.assertEqual(verdict('CORRECT'),'CORRECT')
        self.assertEqual(verdict('Not CORRECT'),'INVALID')
    def test_safe_calculator(self):
        self.assertEqual(calculate('120-80'),'4E+1')
        for expr in ["__import__('os').system('echo unsafe')",'2**10000000','x+2','1/0']:
            with self.assertRaises((ValueError,ArithmeticError)): calculate(expr)
    def test_recalculate_uses_model_expression(self):
        r=recalculate('Calculation: 120 - 80 = 50\nFinal answer: 50')
        self.assertTrue(r['available']);self.assertEqual(r['answer'],'4E+1')
        self.assertFalse(recalculate('120 - 80 = 40\nFinal answer: 55')['available'])
        self.assertFalse(recalculate('Final answer: 50')['available'])
        self.assertFalse(recalculate('x + 2*3 = 6\nFinal answer: 6')['available'])
        self.assertFalse(recalculate('2**3 = 8\nFinal answer: 8')['available'])
        self.assertTrue(recalculate('Profit = $15 - $10 = $4\nFinal answer: 4')['available'])
    def test_no_reference_inputs(self):
        from evaluate import inputs,availability
        r=dict(id='x',question='q',original=dict(decoded_output='Final answer: 50',truncated=False,mean_nll=.1,generated_ids=[1]),verification=dict(verdict='CORRECT',mean_nll=.1),gold='40',label_error=1,actions={'Review':{'success':1}})
        x=inputs([r]);r['gold']='50';r['label_error']=0;r['actions']={'Review':{'success':0}}
        self.assertEqual(x,inputs([r]));self.assertEqual(availability(x).tolist(),[[True,False,True,True]])
if __name__=='__main__': unittest.main()
