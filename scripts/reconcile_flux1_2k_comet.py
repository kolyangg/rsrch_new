"""Complete the old replay receipt when Comet omits .png on suffixed image names."""
import json
import re
from types import SimpleNamespace

from ba_dit.data.manifest import file_hash
from scripts import validate_flux1_2k_fixed96 as replay


if __name__=='__main__':
    assert all((replay.RUN/f'{stage}.done.json').exists() for stage in ('infer','decode','score'))
    original_sub=re.sub
    def normalized_sub(pattern,replacement,value):
        if pattern == r'(?: \(\d+\))?\.png$':
            pattern = r'(?: \(\d+\))?(?:\.png)?$'
        return original_sub(pattern,replacement,value)
    # Keep every immutable historical inference/scoring file unchanged. Only
    # normalize the API's optional extension/duplicate suffix during read-back.
    replay.re=SimpleNamespace(sub=normalized_sub)
    (replay.RUN/'comet_filename_reconciliation.json').write_text(json.dumps({
        'reason':'Comet image filenames may carry (1) without a .png extension',
        'helper_sha256':file_hash(__file__),'inference_or_score_repeated':False},indent=2)+'\n')
    replay.main()
