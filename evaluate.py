"""100 structured policy probes. Not a held-out prediction-of-personality test."""
import json
from pathlib import Path
from model import DecisionModel

ROOT=Path(__file__).resolve().parent

def benchmark():
    rows=[]
    def add(domain,question,contexts,options,expected,status,evidence):
        rows.append(dict(id=f'd{len(rows)+1:03}',domain=domain,question=question,contexts=contexts,
                         options=options,expected_decision=expected,expected_status=status,evidence=evidence))
    unknown='No personal preference supplied. Expected abstention is a system behavior, not a taste label.'
    for domain,sets in {
      'food':[['Sushi','Pizza'],['Falafel','Hummus'],['Coffee','Tea'],['Sweet','Savory'],['Spicy','Mild'],
              ['Steak','Tofu'],['Pasta','Rice'],['Olives','Pickles'],['Cake','Fruit'],['Chocolate','Vanilla'],
              ['Italian','Japanese','Israeli'],['Apple','Banana','Orange'],['Soup','Salad','Sandwich'],['Fish','Chicken','Lentils'],
              ['Espresso','Latte','Tea'],['Breakfast','Lunch','Dinner'],['Bread','Rice','Potato'],['Yogurt','Ice cream','Sorbet'],
              ['Fresh','Grilled','Roasted'],['Noodles','Curry','Tacos']],
      'music':[['Jazz','Rock'],['Bach','Mozart'],['Pop','Hip hop'],['Opera','Electronic'],['Live','Recorded'],
               ['Piano','Guitar'],['Vocals','Instrumental'],['Hebrew','English'],['Reggae','Country'],['Metal','Folk'],
               ['Jazz','Classical','Pop'],['Beatles','Queen','Rolling Stones'],['Bach','Mozart','Beethoven'],['Piano','Violin','Guitar'],
               ['Swift','Beyonce','Adele'],['Concert','Festival','Home listening'],['Blues','Soul','Funk'],['Rock','Metal','Punk'],
               ['Solo','Duet','Orchestra'],['Musical theater','Opera','Ballet']],
      'politics':[['Democrats','Republicans'],['Likud','Yesh Atid'],['Higher taxes','Lower taxes'],['Mandatory voting','Voluntary voting'],
                  ['Proportional voting','District voting'],['Term limits','No term limits'],['More immigration','Less immigration'],
                  ['Carbon tax','No carbon tax'],['Public healthcare','Private healthcare'],['Two state solution','Other solution'],
                  ['Left','Center','Right'],['Climate','Housing','Defense'],['Parliamentary','Presidential','Mixed'],
                  ['More regulation','Less regulation','Current regulation'],['Local','National','Federal'],['Party A','Party B','Party C'],
                  ['Policy A','Policy B','Policy C'],['Candidate A','Candidate B','Candidate C'],['Raise taxes','Keep taxes','Cut taxes'],
                  ['Expand services','Keep services','Reduce services']]
    }.items():
        for choices in sets:
            add(domain,f'Which {domain} option would I prefer?',[domain],
                [dict(id=str(i),label=c,attributes={}) for i,c in enumerate(choices)],None,'insufficient_preference_evidence',unknown)
    destinations=['Australia','United States','United Kingdom','France','Japan','Canada','Germany','Italy','Spain','New Zealand']
    for destination in destinations:
        add('countries',f'Which flight to {destination} would I choose?',['travel','countries'],[
            dict(id='shabbat',label='Fly on Shabbat',attributes={'shabbat_flight':1}),
            dict(id='weekday',label='Weekday flight; possibly miss a conference day',attributes={'avoids_shabbat':1,'misses_one_or_two_days':1})
        ],'weekday','only_feasible_option','Explicit no flying on Shabbat. Destination does not establish a country preference.')
    for i in range(10):
        choices=[destinations[i],destinations[(i+1)%10],destinations[(i+3)%10]]
        add('countries','Which country would I prefer to live in?',['countries'],
            [dict(id=str(j),label=c,attributes={}) for j,c in enumerate(choices)],None,'insufficient_preference_evidence',unknown)
    pairs=[
      ('communication','concise','speculative','Which answer style would I prefer?'),
      ('communication','verified','speculative','Should a reply use verified facts or speculation?'),
      ('communication','autonomous_completion','redundant_approval','Which assistant behavior fits my requests?'),
      ('communication','finished_deliverable','redundant_approval','Should authorized work finish or stop for redundant approval?'),
      ('design','pastel','printed_name','Which portrait matches my revisions?'),
      ('design','coarse_beard','printed_name','Which portrait detail did I request?'),
      ('cv','single_page','multi_page','Which CV length did I request?'),
      ('website','netlify_domain','old_custom_domain','Which production domain should I choose?'),
      ('teaching','merged_pr','unmerged_pr','Which contribution satisfies my course?'),
      ('teaching','required_services','frontend_only','Which project satisfies my course?'),
      ('analytics','genuine_visitors','ci_as_visitors','Which analytics report matches my requirements?'),
      ('food','explicitly_preferred','explicitly_disliked','If I explicitly state A is preferred and B disliked, which should win?'),
      ('music','explicitly_preferred','explicitly_disliked','If I explicitly prefer song A over B, which should win?'),
      ('politics','explicitly_preferred','explicitly_disliked','If I explicitly state a political preference A over B, which should win?'),
      ('countries','explicitly_preferred','explicitly_disliked','If I explicitly prefer destination A over B, which should win?'),
      ('people','explicitly_preferred','explicitly_disliked','If I explicitly prefer collaborator A over B, which should win?'),
    ]
    for context,good,bad,question in pairs:
        hypothetical=good=='explicitly_preferred'
        add('work',question,[context],[dict(id='A',attributes={good:1}),dict(id='B',attributes={bad:1})],
            'A','decision','Hypothetical supplied preference, not a Yossi fact.' if hypothetical else 'Documented qualitative preference; numerical utility authored, not measured.')
    for choices in [['OpenAI','Anthropic','NVIDIA'],['Modal','Daytona','AWS'],['Person A','Person B','Person C'],['Colleague A','Colleague B']]:
        add('work','Which company or person would I choose?',['career','people'],
            [dict(id=str(i),label=c,attributes={}) for i,c in enumerate(choices)],None,'insufficient_preference_evidence',unknown)
    assert len(rows)==100
    return rows

def run():
    model=DecisionModel()
    rows=benchmark()
    results=[]
    for row in rows:
        out=model.decide(row['question'],row['contexts'],row['options'])
        results.append(dict(row,output=out,policy_check_passed=(out['decision']==row['expected_decision'] and out['status']==row['expected_status'])))
    summary={'total_probes':100,'total_parameters':model.parameter_count,
             'policy_checks_passed':sum(r['policy_check_passed'] for r in results),
             'decisions':sum(r['output']['decision'] is not None for r in results),
             'abstentions':sum(r['output']['decision'] is None for r in results),
             'independent_user_labeled_predictions':0,
             'meaning':'Rule-conformance and abstention tests with structured inputs; not 100% accuracy at predicting a person.',
             'domains':{d:sum(r['domain']==d for r in results) for d in ['food','countries','music','politics','work']}}
    (ROOT/'benchmark_100.json').write_text(json.dumps(rows,indent=2)+'\n')
    (ROOT/'evaluation_100.json').write_text(json.dumps(dict(summary,results=results),indent=2)+'\n')
    lines=['# Evaluation: weight-space decision model','',
           f"{summary['policy_checks_passed']}/100 policy checks passed: {summary['decisions']} decisions, {summary['abstentions']} abstentions.",
           '', '**This is not predictive personality accuracy.** Inputs explicitly encode the relevant option attributes. Expected results are rule-conformance targets. Five work probes use hypothetical caller-supplied preferences. No independent user-labeled predictions have been evaluated.',
           '', 'There are 20 cases each for food, countries/travel, music, politics, and work/companies/people. Unstated tastes produce abstention. General country preferences are unknown; travel constraints are known.',
           '', 'The separate [gradient baseline](baselines/gradient/EVALUATION.md) tested 100 raw-text questions: 15/24 supported answers correct, 76 without ground truth. It is a different task/input contract and not an apples-to-apples accuracy comparison.',
           '', '| ID | Domain | Question | Decision | Status | Check |','|---|---|---|---|---|---|']
    for r in results:
        lines.append(f"| {r['id']} | {r['domain']} | {r['question']} | {r['output']['decision']} | {r['output']['status']} | {'pass' if r['policy_check_passed'] else 'FAIL'} |")
    (ROOT/'EVALUATION.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': run()
