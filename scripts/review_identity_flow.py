"""Review generated identity-flow panels with the correct model and Comet metadata."""
import argparse
from pathlib import Path
from scripts.identity_face_flow import experiment
from scripts.review_reference_refiner import review

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    a=p.parse_args()
    review(a.run.resolve(),experiment_factory=experiment,
           model_label='Three BA reads + identity loss',initial_label='Parent best500')
