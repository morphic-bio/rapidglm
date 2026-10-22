import importlib.util
import pathlib
import unittest
import pandas as pd
spec=importlib.util.spec_from_file_location("pilot",pathlib.Path(__file__).resolve().parents[1]/"scripts/replogle_pilot.py")
pilot=importlib.util.module_from_spec(spec); spec.loader.exec_module(pilot)
class Replogle(unittest.TestCase):
    def test_common_negatives_include_effect_statuses(self):
        r=pd.DataFrame(dict(response_id=["a","b","c"],target=["x"]*3,
                            status=["ok","ok_effect_failed","fit_failed"],p_value=[.01,.5,.02]))
        r=r[r.status.str.startswith("ok")]
        s=pd.DataFrame(dict(response_id=["b","c"],grna_target=["x"]*2,p_value=[.2,.03]))
        paired=pilot.common_negative_pairs(r,s)
        self.assertEqual(paired.response_id.tolist(),["b"])
if __name__=="__main__": unittest.main()
