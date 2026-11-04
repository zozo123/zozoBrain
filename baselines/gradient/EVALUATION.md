# 100-question evaluation

Executed all 100 items. 15/24 labeled items correct; 76 have no ground truth and are not scored.

Labels are assistant interpretations of explicit history or stated professional fit, not user-confirmed answers. No food, music, political or general country-preference labels were invented.

| Domain | Executed | Labeled | Correct | Accuracy |
|---|---:|---:|---:|---:|
| food | 20 | 0 | 0 | N/A |
| countries | 20 | 7 | 4 | 57.1% |
| music | 20 | 0 | 0 | N/A |
| politics | 20 | 0 | 0 | N/A |
| work | 20 | 17 | 11 | 64.7% |

Scores are not calibrated. An apparently strong score on an unlabeled item is not evidence about Yossi. These items expose the need for personal labels.

Full inputs, labels, provenance, scores, and outputs: `evaluation_100.json`. Benchmark never participates in training.

## All predictions

| ID | Domain | Question | Prediction | Result |
|---|---|---|---|---|
| q001 | food | Do I enjoy eating sushi? | yes | unlabeled |
| q002 | food | Do I enjoy eating hummus? | yes | unlabeled |
| q003 | food | Do I enjoy eating pizza? | yes | unlabeled |
| q004 | food | Do I enjoy eating falafel? | yes | unlabeled |
| q005 | food | Do I enjoy eating steak? | yes | unlabeled |
| q006 | food | Do I enjoy eating spicy food? | yes | unlabeled |
| q007 | food | Do I enjoy eating chocolate? | yes | unlabeled |
| q008 | food | Do I enjoy eating olives? | yes | unlabeled |
| q009 | food | Do I enjoy eating seafood? | yes | unlabeled |
| q010 | food | Do I enjoy eating vegan meals? | yes | unlabeled |
| q011 | food | Which food or drink would I prefer? | Coffee > Tea | unlabeled |
| q012 | food | Which food or drink would I prefer? | Savory breakfast > Sweet breakfast | unlabeled |
| q013 | food | Which food or drink would I prefer? | Sushi > Pizza | unlabeled |
| q014 | food | Which food or drink would I prefer? | Falafel > Hummus | unlabeled |
| q015 | food | Which food or drink would I prefer? | Mild food > Spicy food | unlabeled |
| q016 | food | Which food or drink would I prefer? | Israeli food > Italian food > Japanese food | unlabeled |
| q017 | food | Which food or drink would I prefer? | Apples > Oranges > Bananas | unlabeled |
| q018 | food | Which food or drink would I prefer? | Ice cream > Fruit > Cake | unlabeled |
| q019 | food | Which food or drink would I prefer? | Pasta > Rice > Potatoes | unlabeled |
| q020 | food | Which food or drink would I prefer? | Breakfast > Lunch > Dinner | unlabeled |
| q021 | countries | Would I take a Saturday flight to Australia? | no | correct |
| q022 | countries | Is departing from San Francisco an option for my Australia trip? | no | incorrect |
| q023 | countries | Would I shorten conference attendance to fit my travel constraints? | no | incorrect |
| q024 | countries | Must an Australia conference trip last exactly ten days? | no | correct |
| q025 | countries | Am I open to a career move to San Francisco in the United States? | yes | correct |
| q026 | countries | Do I prefer Australia over the United States? | no | unlabeled |
| q027 | countries | Is Japan my favorite country to visit? | no | unlabeled |
| q028 | countries | Would I rather live in Canada than Israel? | yes | unlabeled |
| q029 | countries | Do I dislike traveling to the United Kingdom? | no | unlabeled |
| q030 | countries | Is Italy my favorite holiday destination? | no | unlabeled |
| q031 | countries | Which flight would I take to another country? | Saturday daytime flight > Sunday flight | incorrect |
| q032 | countries | Which conference travel plan fits me? | Miss one day and avoid Shabbat flights > Attend every day by flying on Shabbat | correct |
| q033 | countries | Which country would I personally prefer to visit? | Israel > United States | unlabeled |
| q034 | countries | Which country would I personally prefer to visit? | New Zealand > Australia | unlabeled |
| q035 | countries | Which country would I personally prefer to visit? | France > Italy | unlabeled |
| q036 | countries | Which country would I personally prefer to visit? | South Korea > Japan > Thailand | unlabeled |
| q037 | countries | Which country would I personally prefer to visit? | United Kingdom > Germany > Canada | unlabeled |
| q038 | countries | Which country would I personally prefer to visit? | Spain > Greece > Portugal | unlabeled |
| q039 | countries | Which country would I personally prefer to visit? | Sweden > Norway > Denmark | unlabeled |
| q040 | countries | Which country would I personally prefer to visit? | Israel > United States > Australia | unlabeled |
| q041 | music | Do I enjoy listening to jazz? | no | unlabeled |
| q042 | music | Do I enjoy listening to classical music? | yes | unlabeled |
| q043 | music | Do I enjoy listening to heavy metal? | yes | unlabeled |
| q044 | music | Do I enjoy listening to hip hop? | no | unlabeled |
| q045 | music | Do I enjoy listening to Israeli pop? | no | unlabeled |
| q046 | music | Do I enjoy listening to country music? | no | unlabeled |
| q047 | music | Do I enjoy listening to electronic music? | yes | unlabeled |
| q048 | music | Do I enjoy listening to opera? | no | unlabeled |
| q049 | music | Do I enjoy listening to reggae? | yes | unlabeled |
| q050 | music | Do I enjoy listening to musical theater? | no | unlabeled |
| q051 | music | Which music option would I enjoy more? | Jazz > Rock | unlabeled |
| q052 | music | Which music option would I enjoy more? | Mozart > Bach | unlabeled |
| q053 | music | Which music option would I enjoy more? | Taylor Swift > Beyonce | unlabeled |
| q054 | music | Which music option would I enjoy more? | Studio recording > Live concert | unlabeled |
| q055 | music | Which music option would I enjoy more? | Vocal music > Instrumental music | unlabeled |
| q056 | music | Which music option would I enjoy more? | Jazz > Pop > Classical | unlabeled |
| q057 | music | Which music option would I enjoy more? | Rolling Stones > Queen > Beatles | unlabeled |
| q058 | music | Which music option would I enjoy more? | Piano > Guitar > Violin | unlabeled |
| q059 | music | Which music option would I enjoy more? | English songs > Hebrew songs > Spanish songs | unlabeled |
| q060 | music | Which music option would I enjoy more? | Electronic music > Hip hop > Opera | unlabeled |
| q061 | politics | Do I support the Democratic Party? | yes | unlabeled |
| q062 | politics | Do I support the Republican Party? | yes | unlabeled |
| q063 | politics | Do I support Likud? | yes | unlabeled |
| q064 | politics | Do I favor a two state solution? | yes | unlabeled |
| q065 | politics | Do I support higher income taxes? | no | unlabeled |
| q066 | politics | Do I favor mandatory voting? | yes | unlabeled |
| q067 | politics | Do I support proportional representation? | yes | unlabeled |
| q068 | politics | Do I favor lower immigration levels? | yes | unlabeled |
| q069 | politics | Do I support a carbon tax? | no | unlabeled |
| q070 | politics | Do I favor term limits for elected leaders? | yes | unlabeled |
| q071 | politics | Which political position would I personally prefer? | Republicans > Democrats | unlabeled |
| q072 | politics | Which political position would I personally prefer? | Yesh Atid > Likud | unlabeled |
| q073 | politics | Which political position would I personally prefer? | Lower taxes and fewer public services > Higher taxes and more public services | unlabeled |
| q074 | politics | Which political position would I personally prefer? | Mandatory voting > Voluntary voting | unlabeled |
| q075 | politics | Which political position would I personally prefer? | Proportional representation > Single member districts | unlabeled |
| q076 | politics | Which political position would I personally prefer? | Right > Center > Left | unlabeled |
| q077 | politics | Which political position would I personally prefer? | Climate policy > Defense policy > Housing policy | unlabeled |
| q078 | politics | Which political position would I personally prefer? | Parliamentary system > Presidential system > Mixed system | unlabeled |
| q079 | politics | Which political position would I personally prefer? | Public healthcare > Mixed healthcare > Private healthcare | unlabeled |
| q080 | politics | Which political position would I personally prefer? | Current immigration levels > Less immigration > More immigration | unlabeled |
| q081 | work | Would I rather receive a short direct response? | yes | correct |
| q082 | work | Should an assistant confidently make up a missing fact? | yes | incorrect |
| q083 | work | Would I want a resume that fits on one page? | no | incorrect |
| q084 | work | Can my course students satisfy the contribution rule with pending PRs? | no | correct |
| q085 | work | Is a frontend alone enough for my EASS project? | yes | incorrect |
| q086 | work | Do I want real visitors distinguished from automated website tests? | yes | correct |
| q087 | work | Should my website move to a new Netlify subdomain? | yes | incorrect |
| q088 | work | Is secure execution for AI agents relevant to my research? | no | incorrect |
| q089 | work | Would I like to compare measured build durations? | yes | correct |
| q090 | work | Should I only showcase repositories with no stars? | no | correct |
| q091 | work | Which company role fits my stated AI career goals? | Unrelated retail cashier > OpenAI reinforcement learning researcher | incorrect |
| q092 | work | Which company role fits my sandbox work? | Daytona sandbox engineer > Fashion shop sales assistant | correct |
| q093 | work | Which companies would I prefer as employers? | NVIDIA > Anthropic > OpenAI | unlabeled |
| q094 | work | Which companies would I prefer for compute? | Modal > Daytona > AWS | unlabeled |
| q095 | work | Which people would I most enjoy spending time with? | Elon Musk > Taylor Swift > Albert Einstein | unlabeled |
| q096 | work | Which person fits a secure execution collaboration? | Engineer with sandbox isolation expertise > Salesperson with no relevant systems experience | correct |
| q097 | work | Which biography link did I request? | incredibuild.com > islo.dev | correct |
| q098 | work | Which production domain did I choose? | yossieliaz.netlify.app > www.yossieliaz.com | correct |
| q099 | work | Which project implementation matches my course? | Backend frontend and third microservice > Frontend without a backend | correct |
| q100 | work | Which response would I prefer? | Concise verified answer > Long speculative answer | correct |
