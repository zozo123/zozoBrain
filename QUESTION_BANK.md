# 100 questions about Yossi Eliaz

Can a tiny decision policy predict a person? These are the questions we want to measure—not claims that we already know all the answers.

**20 questions each:** food, countries/travel, music, politics, and work (including companies and people). The original raw-text baseline ran all 100: 15 of 24 supported labels matched; 76 had no ground truth. Those supported labels are assistant interpretations of history, not a fresh questionnaire personally completed by Yossi.

This page renames first-person questions for readability; it is not a new evaluation run. The exact evaluated texts, options, predictions, scores, and provenance remain in [the raw-text baseline report](baselines/gradient/evaluation_100.json). The main direct-weight model requires context tags and option attributes; these questions alone are not sufficient input to it.

## How to use this bank

Have Yossi answer before changing weights. Keep a separate held-out set, record dates and preference changes, and score only labeled answers. Never count an unknown as correct. When a coding agent maps text to structured attributes, log those attributes separately and report the agent contribution.

## Food

**q001. Does Yossi Eliaz enjoy eating sushi?**

Current reference: **Needs Yossi’s answer**.

**q002. Does Yossi Eliaz enjoy eating hummus?**

Current reference: **Needs Yossi’s answer**.

**q003. Does Yossi Eliaz enjoy eating pizza?**

Current reference: **Needs Yossi’s answer**.

**q004. Does Yossi Eliaz enjoy eating falafel?**

Current reference: **Needs Yossi’s answer**.

**q005. Does Yossi Eliaz enjoy eating steak?**

Current reference: **Needs Yossi’s answer**.

**q006. Does Yossi Eliaz enjoy eating spicy food?**

Current reference: **Needs Yossi’s answer**.

**q007. Does Yossi Eliaz enjoy eating chocolate?**

Current reference: **Needs Yossi’s answer**.

**q008. Does Yossi Eliaz enjoy eating olives?**

Current reference: **Needs Yossi’s answer**.

**q009. Does Yossi Eliaz enjoy eating seafood?**

Current reference: **Needs Yossi’s answer**.

**q010. Does Yossi Eliaz enjoy eating vegan meals?**

Current reference: **Needs Yossi’s answer**.

**q011. Which food or drink would Yossi Eliaz prefer?**

Options: Coffee / Tea.

Current reference: **Needs Yossi’s answer**.

**q012. Which food or drink would Yossi Eliaz prefer?**

Options: Sweet breakfast / Savory breakfast.

Current reference: **Needs Yossi’s answer**.

**q013. Which food or drink would Yossi Eliaz prefer?**

Options: Sushi / Pizza.

Current reference: **Needs Yossi’s answer**.

**q014. Which food or drink would Yossi Eliaz prefer?**

Options: Hummus / Falafel.

Current reference: **Needs Yossi’s answer**.

**q015. Which food or drink would Yossi Eliaz prefer?**

Options: Spicy food / Mild food.

Current reference: **Needs Yossi’s answer**.

**q016. Which food or drink would Yossi Eliaz prefer?**

Options: Italian food / Japanese food / Israeli food.

Current reference: **Needs Yossi’s answer**.

**q017. Which food or drink would Yossi Eliaz prefer?**

Options: Apples / Bananas / Oranges.

Current reference: **Needs Yossi’s answer**.

**q018. Which food or drink would Yossi Eliaz prefer?**

Options: Ice cream / Cake / Fruit.

Current reference: **Needs Yossi’s answer**.

**q019. Which food or drink would Yossi Eliaz prefer?**

Options: Pasta / Rice / Potatoes.

Current reference: **Needs Yossi’s answer**.

**q020. Which food or drink would Yossi Eliaz prefer?**

Options: Breakfast / Lunch / Dinner.

Current reference: **Needs Yossi’s answer**.

## Countries

**q021. Would Yossi Eliaz take a Saturday flight to Australia?**

Current reference: **No**.

**q022. Is departing from San Francisco an option for Yossi Eliaz’s Australia trip?**

Current reference: **Yes**.

**q023. Would Yossi Eliaz shorten conference attendance to fit Yossi Eliaz’s travel constraints?**

Current reference: **Yes**.

**q024. Must an Australia conference trip last exactly ten days?**

Current reference: **No**.

**q025. Is Yossi Eliaz open to a career move to San Francisco in the United States?**

Current reference: **Yes**.

**q026. Does Yossi Eliaz prefer Australia over the United States?**

Current reference: **Needs Yossi’s answer**.

**q027. Is Japan Yossi Eliaz’s favorite country to visit?**

Current reference: **Needs Yossi’s answer**.

**q028. Would Yossi Eliaz rather live in Canada than Israel?**

Current reference: **Needs Yossi’s answer**.

**q029. Does Yossi Eliaz dislike traveling to the United Kingdom?**

Current reference: **Needs Yossi’s answer**.

**q030. Is Italy Yossi Eliaz’s favorite holiday destination?**

Current reference: **Needs Yossi’s answer**.

**q031. Which flight would Yossi Eliaz take to another country?**

Options: Sunday flight / Saturday daytime flight.

Current reference: **Sunday flight > Saturday daytime flight**.

**q032. Which conference travel plan fits Yossi Eliaz?**

Options: Miss one day and avoid Shabbat flights / Attend every day by flying on Shabbat.

Current reference: **Miss one day and avoid Shabbat flights > Attend every day by flying on Shabbat**.

**q033. Which country would Yossi Eliaz personally prefer to visit?**

Options: Israel / United States.

Current reference: **Needs Yossi’s answer**.

**q034. Which country would Yossi Eliaz personally prefer to visit?**

Options: Australia / New Zealand.

Current reference: **Needs Yossi’s answer**.

**q035. Which country would Yossi Eliaz personally prefer to visit?**

Options: France / Italy.

Current reference: **Needs Yossi’s answer**.

**q036. Which country would Yossi Eliaz personally prefer to visit?**

Options: Japan / South Korea / Thailand.

Current reference: **Needs Yossi’s answer**.

**q037. Which country would Yossi Eliaz personally prefer to visit?**

Options: Canada / Germany / United Kingdom.

Current reference: **Needs Yossi’s answer**.

**q038. Which country would Yossi Eliaz personally prefer to visit?**

Options: Spain / Portugal / Greece.

Current reference: **Needs Yossi’s answer**.

**q039. Which country would Yossi Eliaz personally prefer to visit?**

Options: Norway / Sweden / Denmark.

Current reference: **Needs Yossi’s answer**.

**q040. Which country would Yossi Eliaz personally prefer to visit?**

Options: Israel / Australia / United States.

Current reference: **Needs Yossi’s answer**.

## Music

**q041. Does Yossi Eliaz enjoy listening to jazz?**

Current reference: **Needs Yossi’s answer**.

**q042. Does Yossi Eliaz enjoy listening to classical music?**

Current reference: **Needs Yossi’s answer**.

**q043. Does Yossi Eliaz enjoy listening to heavy metal?**

Current reference: **Needs Yossi’s answer**.

**q044. Does Yossi Eliaz enjoy listening to hip hop?**

Current reference: **Needs Yossi’s answer**.

**q045. Does Yossi Eliaz enjoy listening to Israeli pop?**

Current reference: **Needs Yossi’s answer**.

**q046. Does Yossi Eliaz enjoy listening to country music?**

Current reference: **Needs Yossi’s answer**.

**q047. Does Yossi Eliaz enjoy listening to electronic music?**

Current reference: **Needs Yossi’s answer**.

**q048. Does Yossi Eliaz enjoy listening to opera?**

Current reference: **Needs Yossi’s answer**.

**q049. Does Yossi Eliaz enjoy listening to reggae?**

Current reference: **Needs Yossi’s answer**.

**q050. Does Yossi Eliaz enjoy listening to musical theater?**

Current reference: **Needs Yossi’s answer**.

**q051. Which music option would Yossi Eliaz enjoy more?**

Options: Jazz / Rock.

Current reference: **Needs Yossi’s answer**.

**q052. Which music option would Yossi Eliaz enjoy more?**

Options: Bach / Mozart.

Current reference: **Needs Yossi’s answer**.

**q053. Which music option would Yossi Eliaz enjoy more?**

Options: Taylor Swift / Beyonce.

Current reference: **Needs Yossi’s answer**.

**q054. Which music option would Yossi Eliaz enjoy more?**

Options: Live concert / Studio recording.

Current reference: **Needs Yossi’s answer**.

**q055. Which music option would Yossi Eliaz enjoy more?**

Options: Instrumental music / Vocal music.

Current reference: **Needs Yossi’s answer**.

**q056. Which music option would Yossi Eliaz enjoy more?**

Options: Jazz / Classical / Pop.

Current reference: **Needs Yossi’s answer**.

**q057. Which music option would Yossi Eliaz enjoy more?**

Options: Beatles / Rolling Stones / Queen.

Current reference: **Needs Yossi’s answer**.

**q058. Which music option would Yossi Eliaz enjoy more?**

Options: Piano / Violin / Guitar.

Current reference: **Needs Yossi’s answer**.

**q059. Which music option would Yossi Eliaz enjoy more?**

Options: Hebrew songs / English songs / Spanish songs.

Current reference: **Needs Yossi’s answer**.

**q060. Which music option would Yossi Eliaz enjoy more?**

Options: Opera / Hip hop / Electronic music.

Current reference: **Needs Yossi’s answer**.

## Politics

**q061. Does Yossi Eliaz support the Democratic Party?**

Current reference: **Needs Yossi’s answer**.

**q062. Does Yossi Eliaz support the Republican Party?**

Current reference: **Needs Yossi’s answer**.

**q063. Does Yossi Eliaz support Likud?**

Current reference: **Needs Yossi’s answer**.

**q064. Does Yossi Eliaz favor a two state solution?**

Current reference: **Needs Yossi’s answer**.

**q065. Does Yossi Eliaz support higher income taxes?**

Current reference: **Needs Yossi’s answer**.

**q066. Does Yossi Eliaz favor mandatory voting?**

Current reference: **Needs Yossi’s answer**.

**q067. Does Yossi Eliaz support proportional representation?**

Current reference: **Needs Yossi’s answer**.

**q068. Does Yossi Eliaz favor lower immigration levels?**

Current reference: **Needs Yossi’s answer**.

**q069. Does Yossi Eliaz support a carbon tax?**

Current reference: **Needs Yossi’s answer**.

**q070. Does Yossi Eliaz favor term limits for elected leaders?**

Current reference: **Needs Yossi’s answer**.

**q071. Which political position would Yossi Eliaz personally prefer?**

Options: Democrats / Republicans.

Current reference: **Needs Yossi’s answer**.

**q072. Which political position would Yossi Eliaz personally prefer?**

Options: Likud / Yesh Atid.

Current reference: **Needs Yossi’s answer**.

**q073. Which political position would Yossi Eliaz personally prefer?**

Options: Higher taxes and more public services / Lower taxes and fewer public services.

Current reference: **Needs Yossi’s answer**.

**q074. Which political position would Yossi Eliaz personally prefer?**

Options: Mandatory voting / Voluntary voting.

Current reference: **Needs Yossi’s answer**.

**q075. Which political position would Yossi Eliaz personally prefer?**

Options: Proportional representation / Single member districts.

Current reference: **Needs Yossi’s answer**.

**q076. Which political position would Yossi Eliaz personally prefer?**

Options: Left / Center / Right.

Current reference: **Needs Yossi’s answer**.

**q077. Which political position would Yossi Eliaz personally prefer?**

Options: Climate policy / Housing policy / Defense policy.

Current reference: **Needs Yossi’s answer**.

**q078. Which political position would Yossi Eliaz personally prefer?**

Options: Parliamentary system / Presidential system / Mixed system.

Current reference: **Needs Yossi’s answer**.

**q079. Which political position would Yossi Eliaz personally prefer?**

Options: Public healthcare / Private healthcare / Mixed healthcare.

Current reference: **Needs Yossi’s answer**.

**q080. Which political position would Yossi Eliaz personally prefer?**

Options: More immigration / Current immigration levels / Less immigration.

Current reference: **Needs Yossi’s answer**.

## Work

**q081. Would Yossi Eliaz rather receive a short direct response?**

Current reference: **Yes**.

**q082. Should an assistant confidently make up a missing fact?**

Current reference: **No**.

**q083. Would Yossi Eliaz want a resume that fits on one page?**

Current reference: **Yes**.

**q084. Can Yossi Eliaz’s course students satisfy the contribution rule with pending PRs?**

Current reference: **No**.

**q085. Is a frontend alone enough for Yossi Eliaz’s EASS project?**

Current reference: **No**.

**q086. Does Yossi Eliaz want real visitors distinguished from automated website tests?**

Current reference: **Yes**.

**q087. Should Yossi Eliaz’s website move to a new Netlify subdomain?**

Current reference: **No**.

**q088. Is secure execution for AI agents relevant to Yossi Eliaz’s research?**

Current reference: **Yes**.

**q089. Would Yossi Eliaz like to compare measured build durations?**

Current reference: **Yes**.

**q090. Should Yossi Eliaz only showcase repositories with no stars?**

Current reference: **No**.

**q091. Which company role fits Yossi Eliaz’s stated AI career goals?**

Options: OpenAI reinforcement learning researcher / Unrelated retail cashier.

Current reference: **OpenAI reinforcement learning researcher > Unrelated retail cashier**.

**q092. Which company role fits Yossi Eliaz’s sandbox work?**

Options: Daytona sandbox engineer / Fashion shop sales assistant.

Current reference: **Daytona sandbox engineer > Fashion shop sales assistant**.

**q093. Which companies would Yossi Eliaz prefer as employers?**

Options: OpenAI / Anthropic / NVIDIA.

Current reference: **Needs Yossi’s answer**.

**q094. Which companies would Yossi Eliaz prefer for compute?**

Options: Modal / Daytona / AWS.

Current reference: **Needs Yossi’s answer**.

**q095. Which people would Yossi Eliaz most enjoy spending time with?**

Options: Taylor Swift / Elon Musk / Albert Einstein.

Current reference: **Needs Yossi’s answer**.

**q096. Which person fits a secure execution collaboration?**

Options: Engineer with sandbox isolation expertise / Salesperson with no relevant systems experience.

Current reference: **Engineer with sandbox isolation expertise > Salesperson with no relevant systems experience**.

**q097. Which biography link did Yossi Eliaz request?**

Options: incredibuild.com / islo.dev.

Current reference: **incredibuild.com > islo.dev**.

**q098. Which production domain did Yossi Eliaz choose?**

Options: yossieliaz.netlify.app / www.yossieliaz.com.

Current reference: **yossieliaz.netlify.app > www.yossieliaz.com**.

**q099. Which project implementation matches Yossi Eliaz’s course?**

Options: Backend frontend and third microservice / Frontend without a backend.

Current reference: **Backend frontend and third microservice > Frontend without a backend**.

**q100. Which response would Yossi Eliaz prefer?**

Options: Concise verified answer / Long speculative answer.

Current reference: **Concise verified answer > Long speculative answer**.

