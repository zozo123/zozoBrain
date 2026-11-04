"""Freeze 100 evaluation items. Never imported by training."""
import json
from model import ROOT

items=[]
def yes(domain, question, expected=None, evidence=None):
    items.append(dict(id=f'q{len(items)+1:03}',domain=domain,mode='yes_no',question=question,
                      expected=expected,evidence=evidence or 'No explicit preference found; user label required.'))
def rank(domain, question, options, expected=None, evidence=None):
    items.append(dict(id=f'q{len(items)+1:03}',domain=domain,mode='choice' if len(options)==2 else 'rank',
                      question=question,options=options,expected=expected,
                      evidence=evidence or 'No explicit preference order found; user label required.'))

for food in ['sushi','hummus','pizza','falafel','steak','spicy food','chocolate','olives','seafood','vegan meals']:
    yes('food',f'Do I enjoy eating {food}?')
for options in [['Coffee','Tea'],['Sweet breakfast','Savory breakfast'],['Sushi','Pizza'],['Hummus','Falafel'],['Spicy food','Mild food'],
                ['Italian food','Japanese food','Israeli food'],['Apples','Bananas','Oranges'],['Ice cream','Cake','Fruit'],['Pasta','Rice','Potatoes'],['Breakfast','Lunch','Dinner']]:
    rank('food','Which food or drink would I prefer?',options)

yes('countries','Would I take a Saturday flight to Australia?',False,'Explicit no flying on Shabbat; daytime Saturday is included.')
yes('countries','Is departing from San Francisco an option for my Australia trip?',True,'Explicit SF on the table statement.')
yes('countries','Would I shorten conference attendance to fit my travel constraints?',True,'Explicit willingness to miss one or two conference days.')
yes('countries','Must an Australia conference trip last exactly ten days?',False,'User challenged unexplained ten day assumption.')
yes('countries','Am I open to a career move to San Francisco in the United States?',True,'Historical explicit relocation willingness; not a prediction of current acceptance.')
yes('countries','Do I prefer Australia over the United States?')
yes('countries','Is Japan my favorite country to visit?')
yes('countries','Would I rather live in Canada than Israel?')
yes('countries','Do I dislike traveling to the United Kingdom?')
yes('countries','Is Italy my favorite holiday destination?')
rank('countries','Which flight would I take to another country?',['Sunday flight','Saturday daytime flight'],['Sunday flight','Saturday daytime flight'],'Explicit no flying on Shabbat.')
rank('countries','Which conference travel plan fits me?',['Miss one day and avoid Shabbat flights','Attend every day by flying on Shabbat'],['Miss one day and avoid Shabbat flights','Attend every day by flying on Shabbat'],'Explicit travel constraint and attendance flexibility.')
for options in [['Israel','United States'],['Australia','New Zealand'],['France','Italy'],['Japan','South Korea','Thailand'],['Canada','Germany','United Kingdom'],['Spain','Portugal','Greece'],['Norway','Sweden','Denmark'],['Israel','Australia','United States']]:
    rank('countries','Which country would I personally prefer to visit?',options)

for music in ['jazz','classical music','heavy metal','hip hop','Israeli pop','country music','electronic music','opera','reggae','musical theater']:
    yes('music',f'Do I enjoy listening to {music}?')
for options in [['Jazz','Rock'],['Bach','Mozart'],['Taylor Swift','Beyonce'],['Live concert','Studio recording'],['Instrumental music','Vocal music'],
                ['Jazz','Classical','Pop'],['Beatles','Rolling Stones','Queen'],['Piano','Violin','Guitar'],['Hebrew songs','English songs','Spanish songs'],['Opera','Hip hop','Electronic music']]:
    rank('music','Which music option would I enjoy more?',options)

for question in ['Do I support the Democratic Party?','Do I support the Republican Party?','Do I support Likud?',
                 'Do I favor a two state solution?','Do I support higher income taxes?','Do I favor mandatory voting?',
                 'Do I support proportional representation?','Do I favor lower immigration levels?',
                 'Do I support a carbon tax?','Do I favor term limits for elected leaders?']:
    yes('politics',question)
for options in [['Democrats','Republicans'],['Likud','Yesh Atid'],['Higher taxes and more public services','Lower taxes and fewer public services'],
                ['Mandatory voting','Voluntary voting'],['Proportional representation','Single member districts'],
                ['Left','Center','Right'],['Climate policy','Housing policy','Defense policy'],['Parliamentary system','Presidential system','Mixed system'],
                ['Public healthcare','Private healthcare','Mixed healthcare'],['More immigration','Current immigration levels','Less immigration']]:
    rank('politics','Which political position would I personally prefer?',options)

yes('work','Would I rather receive a short direct response?',True,'Repeated concise answer requests.')
yes('work','Should an assistant confidently make up a missing fact?',False,'Repeated verification requests.')
yes('work','Would I want a resume that fits on one page?',True,'Explicit CV format.')
yes('work','Can my course students satisfy the contribution rule with pending PRs?',False,'Merged contributions required.')
yes('work','Is a frontend alone enough for my EASS project?',False,'Backend and third service required.')
yes('work','Do I want real visitors distinguished from automated website tests?',True,'Explicit PostHog requirement.')
yes('work','Should my website move to a new Netlify subdomain?',False,'Explicit preserve subdomain instruction.')
yes('work','Is secure execution for AI agents relevant to my research?',True,'Stated technical work.')
yes('work','Would I like to compare measured build durations?',True,'Explicit CI comparison request.')
yes('work','Should I only showcase repositories with no stars?',False,'Explicit pinned repository request.')
rank('work','Which company role fits my stated AI career goals?',['OpenAI reinforcement learning researcher','Unrelated retail cashier'],['OpenAI reinforcement learning researcher','Unrelated retail cashier'],'Role relevance from career targets; not a company-wide preference.')
rank('work','Which company role fits my sandbox work?',['Daytona sandbox engineer','Fashion shop sales assistant'],['Daytona sandbox engineer','Fashion shop sales assistant'],'Role relevance from technical interests.')
rank('work','Which companies would I prefer as employers?',['OpenAI','Anthropic','NVIDIA'])
rank('work','Which companies would I prefer for compute?',['Modal','Daytona','AWS'])
rank('work','Which people would I most enjoy spending time with?',['Taylor Swift','Elon Musk','Albert Einstein'])
rank('work','Which person fits a secure execution collaboration?',['Engineer with sandbox isolation expertise','Salesperson with no relevant systems experience'],['Engineer with sandbox isolation expertise','Salesperson with no relevant systems experience'],'Inferred professional relevance only.')
rank('work','Which biography link did I request?',['incredibuild.com','islo.dev'],['incredibuild.com','islo.dev'],'Explicit bio revision.')
rank('work','Which production domain did I choose?',['yossieliaz.netlify.app','www.yossieliaz.com'],['yossieliaz.netlify.app','www.yossieliaz.com'],'Explicit production domain.')
rank('work','Which project implementation matches my course?',['Backend frontend and third microservice','Frontend without a backend'],['Backend frontend and third microservice','Frontend without a backend'],'Explicit course requirements.')
rank('work','Which response would I prefer?',['Concise verified answer','Long speculative answer'],['Concise verified answer','Long speculative answer'],'Repeated communication preferences.')

assert len(items)==100
assert all(sum(r['domain']==d for r in items)==20 for d in ['food','countries','music','politics','work'])
(ROOT/'benchmark_100.json').write_text(json.dumps(items,indent=2)+'\n')
