"""Direct weight-space programming: numerical assignments, no fitting or optimizer.

Every dense tensor entry counts toward the budget, including zero entries.
The constants below are authored policy hypotheses, not estimated personal utilities.
"""
import numpy as np

CONTEXTS = ['communication','travel','career','security','research','design','cv','website',
            'teaching','analytics','food','music','politics','countries','people','general']
# name, activation context (None = all contexts), utility weight, veto flag
RULES = [
 ('concise','communication',3,False),
 ('verified','communication',4,False),
 ('speculative','communication',-5,False),
 ('autonomous_completion','communication',3,False),
 ('redundant_approval','communication',-2,False),
 ('finished_deliverable','communication',3,False),
 ('shabbat_flight','travel',-20,True),
 ('avoids_shabbat','travel',3,False),
 ('misses_one_or_two_days','travel',0,False),
 ('scientific_ai','career',4,False),
 ('secure_sandbox','security',4,False),
 ('reproducible','research',3,False),
 ('measured_performance','research',3,False),
 ('open_source','career',2,False),
 ('relevant_expertise','people',3,False),
 ('pastel','design',2,False),
 ('coarse_beard','design',2,False),
 ('printed_name','design',-3,False),
 ('single_page','cv',3,False),
 ('multi_page','cv',-3,False),
 ('netlify_domain','website',4,False),
 ('old_custom_domain','website',-4,False),
 ('merged_pr','teaching',3,False),
 ('unmerged_pr','teaching',-3,False),
 ('required_services','teaching',3,False),
 ('frontend_only','teaching',-3,False),
 ('genuine_visitors','analytics',3,False),
 ('ci_as_visitors','analytics',-3,False),
 ('explicitly_preferred',None,5,False),
 ('explicitly_disliked',None,-5,False),
 ('explicitly_disallowed',None,-20,True),
 ('explicitly_allowed',None,0,False),
]
ATTRIBUTES=[r[0] for r in RULES]

def authored_weights():
    w1=np.zeros((48,48),dtype=np.float32)
    b1=np.zeros(48,dtype=np.float32)
    w2=np.zeros((48,3),dtype=np.float32)
    b2=np.zeros(3,dtype=np.float32)
    for i,(_,context,utility,veto) in enumerate(RULES):
        w1[i,i]=1
        if context is not None:
            w1[32+CONTEXTS.index(context),i]=1
            b1[i]=-1
        w2[i,0]=utility
        w2[i,1]=int(veto)
        # Zero-utility permissibility does not provide an ordering preference.
        w2[i,2]=int(utility!=0 or veto)
    # 16 hidden units reserved, all zero. They still count in total parameters.
    return w1,b1,w2,b2

def forward(x,weights=None):
    w1,b1,w2,b2=authored_weights() if weights is None else weights
    return np.maximum(0,x@w1+b1)@w2+b2

if __name__=='__main__':
    from pathlib import Path
    tensors=authored_weights()
    np.savez(Path(__file__).with_name('weights.npz'),**dict(zip(['w1','b1','w2','b2'],tensors)))
    print({'total_parameters':sum(a.size for a in tensors),'nonzero_parameters':sum(np.count_nonzero(a) for a in tensors)})
