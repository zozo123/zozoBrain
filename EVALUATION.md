# Evaluation: weight-space decision model

100/100 policy checks passed: 26 decisions, 74 abstentions.

**This is not predictive personality accuracy.** Inputs explicitly encode the relevant option attributes. Expected results are rule-conformance targets. Five work probes use hypothetical caller-supplied preferences. No independent user-labeled predictions have been evaluated.

There are 20 cases each for food, countries/travel, music, politics, and work/companies/people. Unstated tastes produce abstention. General country preferences are unknown; travel constraints are known.

The separate [gradient baseline](baselines/gradient/EVALUATION.md) tested 100 raw-text questions: 15/24 supported answers correct, 76 without ground truth. It is a different task/input contract and not an apples-to-apples accuracy comparison.

| ID | Domain | Question | Decision | Status | Check |
|---|---|---|---|---|---|
| d001 | food | Which food option would I prefer? | None | insufficient_preference_evidence | pass |
| d002 | food | Which food option would I prefer? | None | insufficient_preference_evidence | pass |
| d003 | food | Which food option would I prefer? | None | insufficient_preference_evidence | pass |
| d004 | food | Which food option would I prefer? | None | insufficient_preference_evidence | pass |
| d005 | food | Which food option would I prefer? | None | insufficient_preference_evidence | pass |
| d006 | food | Which food option would I prefer? | None | insufficient_preference_evidence | pass |
| d007 | food | Which food option would I prefer? | None | insufficient_preference_evidence | pass |
| d008 | food | Which food option would I prefer? | None | insufficient_preference_evidence | pass |
| d009 | food | Which food option would I prefer? | None | insufficient_preference_evidence | pass |
| d010 | food | Which food option would I prefer? | None | insufficient_preference_evidence | pass |
| d011 | food | Which food option would I prefer? | None | insufficient_preference_evidence | pass |
| d012 | food | Which food option would I prefer? | None | insufficient_preference_evidence | pass |
| d013 | food | Which food option would I prefer? | None | insufficient_preference_evidence | pass |
| d014 | food | Which food option would I prefer? | None | insufficient_preference_evidence | pass |
| d015 | food | Which food option would I prefer? | None | insufficient_preference_evidence | pass |
| d016 | food | Which food option would I prefer? | None | insufficient_preference_evidence | pass |
| d017 | food | Which food option would I prefer? | None | insufficient_preference_evidence | pass |
| d018 | food | Which food option would I prefer? | None | insufficient_preference_evidence | pass |
| d019 | food | Which food option would I prefer? | None | insufficient_preference_evidence | pass |
| d020 | food | Which food option would I prefer? | None | insufficient_preference_evidence | pass |
| d021 | music | Which music option would I prefer? | None | insufficient_preference_evidence | pass |
| d022 | music | Which music option would I prefer? | None | insufficient_preference_evidence | pass |
| d023 | music | Which music option would I prefer? | None | insufficient_preference_evidence | pass |
| d024 | music | Which music option would I prefer? | None | insufficient_preference_evidence | pass |
| d025 | music | Which music option would I prefer? | None | insufficient_preference_evidence | pass |
| d026 | music | Which music option would I prefer? | None | insufficient_preference_evidence | pass |
| d027 | music | Which music option would I prefer? | None | insufficient_preference_evidence | pass |
| d028 | music | Which music option would I prefer? | None | insufficient_preference_evidence | pass |
| d029 | music | Which music option would I prefer? | None | insufficient_preference_evidence | pass |
| d030 | music | Which music option would I prefer? | None | insufficient_preference_evidence | pass |
| d031 | music | Which music option would I prefer? | None | insufficient_preference_evidence | pass |
| d032 | music | Which music option would I prefer? | None | insufficient_preference_evidence | pass |
| d033 | music | Which music option would I prefer? | None | insufficient_preference_evidence | pass |
| d034 | music | Which music option would I prefer? | None | insufficient_preference_evidence | pass |
| d035 | music | Which music option would I prefer? | None | insufficient_preference_evidence | pass |
| d036 | music | Which music option would I prefer? | None | insufficient_preference_evidence | pass |
| d037 | music | Which music option would I prefer? | None | insufficient_preference_evidence | pass |
| d038 | music | Which music option would I prefer? | None | insufficient_preference_evidence | pass |
| d039 | music | Which music option would I prefer? | None | insufficient_preference_evidence | pass |
| d040 | music | Which music option would I prefer? | None | insufficient_preference_evidence | pass |
| d041 | politics | Which politics option would I prefer? | None | insufficient_preference_evidence | pass |
| d042 | politics | Which politics option would I prefer? | None | insufficient_preference_evidence | pass |
| d043 | politics | Which politics option would I prefer? | None | insufficient_preference_evidence | pass |
| d044 | politics | Which politics option would I prefer? | None | insufficient_preference_evidence | pass |
| d045 | politics | Which politics option would I prefer? | None | insufficient_preference_evidence | pass |
| d046 | politics | Which politics option would I prefer? | None | insufficient_preference_evidence | pass |
| d047 | politics | Which politics option would I prefer? | None | insufficient_preference_evidence | pass |
| d048 | politics | Which politics option would I prefer? | None | insufficient_preference_evidence | pass |
| d049 | politics | Which politics option would I prefer? | None | insufficient_preference_evidence | pass |
| d050 | politics | Which politics option would I prefer? | None | insufficient_preference_evidence | pass |
| d051 | politics | Which politics option would I prefer? | None | insufficient_preference_evidence | pass |
| d052 | politics | Which politics option would I prefer? | None | insufficient_preference_evidence | pass |
| d053 | politics | Which politics option would I prefer? | None | insufficient_preference_evidence | pass |
| d054 | politics | Which politics option would I prefer? | None | insufficient_preference_evidence | pass |
| d055 | politics | Which politics option would I prefer? | None | insufficient_preference_evidence | pass |
| d056 | politics | Which politics option would I prefer? | None | insufficient_preference_evidence | pass |
| d057 | politics | Which politics option would I prefer? | None | insufficient_preference_evidence | pass |
| d058 | politics | Which politics option would I prefer? | None | insufficient_preference_evidence | pass |
| d059 | politics | Which politics option would I prefer? | None | insufficient_preference_evidence | pass |
| d060 | politics | Which politics option would I prefer? | None | insufficient_preference_evidence | pass |
| d061 | countries | Which flight to Australia would I choose? | weekday | only_feasible_option | pass |
| d062 | countries | Which flight to United States would I choose? | weekday | only_feasible_option | pass |
| d063 | countries | Which flight to United Kingdom would I choose? | weekday | only_feasible_option | pass |
| d064 | countries | Which flight to France would I choose? | weekday | only_feasible_option | pass |
| d065 | countries | Which flight to Japan would I choose? | weekday | only_feasible_option | pass |
| d066 | countries | Which flight to Canada would I choose? | weekday | only_feasible_option | pass |
| d067 | countries | Which flight to Germany would I choose? | weekday | only_feasible_option | pass |
| d068 | countries | Which flight to Italy would I choose? | weekday | only_feasible_option | pass |
| d069 | countries | Which flight to Spain would I choose? | weekday | only_feasible_option | pass |
| d070 | countries | Which flight to New Zealand would I choose? | weekday | only_feasible_option | pass |
| d071 | countries | Which country would I prefer to live in? | None | insufficient_preference_evidence | pass |
| d072 | countries | Which country would I prefer to live in? | None | insufficient_preference_evidence | pass |
| d073 | countries | Which country would I prefer to live in? | None | insufficient_preference_evidence | pass |
| d074 | countries | Which country would I prefer to live in? | None | insufficient_preference_evidence | pass |
| d075 | countries | Which country would I prefer to live in? | None | insufficient_preference_evidence | pass |
| d076 | countries | Which country would I prefer to live in? | None | insufficient_preference_evidence | pass |
| d077 | countries | Which country would I prefer to live in? | None | insufficient_preference_evidence | pass |
| d078 | countries | Which country would I prefer to live in? | None | insufficient_preference_evidence | pass |
| d079 | countries | Which country would I prefer to live in? | None | insufficient_preference_evidence | pass |
| d080 | countries | Which country would I prefer to live in? | None | insufficient_preference_evidence | pass |
| d081 | work | Which answer style would I prefer? | A | decision | pass |
| d082 | work | Should a reply use verified facts or speculation? | A | decision | pass |
| d083 | work | Which assistant behavior fits my requests? | A | decision | pass |
| d084 | work | Should authorized work finish or stop for redundant approval? | A | decision | pass |
| d085 | work | Which portrait matches my revisions? | A | decision | pass |
| d086 | work | Which portrait detail did I request? | A | decision | pass |
| d087 | work | Which CV length did I request? | A | decision | pass |
| d088 | work | Which production domain should I choose? | A | decision | pass |
| d089 | work | Which contribution satisfies my course? | A | decision | pass |
| d090 | work | Which project satisfies my course? | A | decision | pass |
| d091 | work | Which analytics report matches my requirements? | A | decision | pass |
| d092 | work | If I explicitly state A is preferred and B disliked, which should win? | A | decision | pass |
| d093 | work | If I explicitly prefer song A over B, which should win? | A | decision | pass |
| d094 | work | If I explicitly state a political preference A over B, which should win? | A | decision | pass |
| d095 | work | If I explicitly prefer destination A over B, which should win? | A | decision | pass |
| d096 | work | If I explicitly prefer collaborator A over B, which should win? | A | decision | pass |
| d097 | work | Which company or person would I choose? | None | insufficient_preference_evidence | pass |
| d098 | work | Which company or person would I choose? | None | insufficient_preference_evidence | pass |
| d099 | work | Which company or person would I choose? | None | insufficient_preference_evidence | pass |
| d100 | work | Which company or person would I choose? | None | insufficient_preference_evidence | pass |
