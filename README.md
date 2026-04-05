1. download the original analysis reports from database: month level, group by industry
2. follow the numbet on the industry list, paste the number
3. go to 00risk_dic.py calling deepseek api to generate risk dictionaries for you
4. go to 00stop_words.py to generate corresponding stop words
5. go to 00merge_dic to merge dics and stop words
6. go to 01training to generate topics and labels per industry
7. go to !02trans_topics
8. go to 02label_exposure to get firm level risk intensity scores (has a specific name)
9. go to 03LabelCluster get overall risk types
10. go to 04monthly_risk_trend to get the overall risk intensity per firm
