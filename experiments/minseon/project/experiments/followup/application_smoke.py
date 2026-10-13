"""Check reference-free deployment reproduces frozen evaluation predictions."""
import sys,subprocess
from pathlib import Path
import numpy as np
from learn import RUN,read,dump,inputs
from evaluate import ACTIONS

def main():
 saved=read(RUN/'test_predictions.json')['n1000_seed101'];ids=saved['ids'][:5];rows=[read(RUN/'records'/f'{rid}.json') for rid in ids];public=inputs(rows);dump(RUN/'application_smoke_inputs.json',public)
 code=Path(__file__).resolve().parent/'apply.py';subprocess.run([sys.executable,str(code),'--inputs',str(RUN/'application_smoke_inputs.json'),'--hidden-dir',str(RUN/'hidden'),'--output',str(RUN/'application_smoke_predictions.json')],check=True)
 actual=read(RUN/'application_smoke_predictions.json');selection=read(RUN/'test_results.json')['n1000_seed101']['chosen']
 for variant,(aa,bb) in selection.items():
  np.testing.assert_allclose(actual['predictions'][variant]['error_probability'],saved['A'][aa][:5],rtol=1e-10,atol=1e-10)
  np.testing.assert_allclose(actual['predictions'][variant]['action_success_probabilities'],saved['B_probabilities'][bb][:5],rtol=1e-10,atol=1e-10)
  for policy in ['selector','conservative','trigger']:assert actual['predictions'][variant]['choices'][policy]==[ACTIONS[j] for j in saved['choices'][variant+'_'+policy][:5]]
 dump(RUN/'application_smoke_check.json',dict(passed=True,n=5,reference_free_inputs=True,all_A_B_probabilities_and_choices_match=True))
 print('Reference-free application matches evaluation',flush=True)
if __name__=='__main__':main()
