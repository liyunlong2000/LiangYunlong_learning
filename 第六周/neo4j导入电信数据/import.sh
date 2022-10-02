
$NEO4J_HOME/bin/neo4j-admin import \
--database testnode.db \
--nodes=account=header/node1_head.csv,csv/node1.csv \
--nodes=bankcard=header/node2_head.csv,csv/node2.csv \
--nodes=company=header/node3_head.csv,csv/node3.csv \
--nodes=credit=header/node4_head.csv,csv/node4.csv \
--nodes=device=header/node5_head.csv,csv/node5.csv \
--nodes=idcard=header/node6_head.csv,csv/node6.csv \
--nodes=ip=header/node7_head.csv,csv/node7.csv \
--nodes=operator=header/node8_head.csv,csv/node8.csv \
--nodes=wifi=header/node9_head.csv,csv/node9.csv \
--nodes=withdraw=header/node10_head.csv,csv/node10.csv \
--relationships=R_account_2_account=header/rel1_head.csv,csv/rel1.csv \
--relationships=R_account_2_bankcard=header/rel2_head.csv,csv/rel2.csv \
--relationships=R_account_2_credit=header/rel3_head.csv,csv/rel3.csv \
--relationships=R_account_2_device=header/rel4_head.csv,csv/rel4.csv \
--relationships=R_account_2_idcard=header/rel5_head.csv,csv/rel5.csv \
--relationships=R_account_2_ip=header/rel6_head.csv,csv/rel6.csv \
--relationships=R_account_2_wifi=header/rel7_head.csv,csv/rel7.csv \
--relationships=R_account_2_withdraw=header/rel8_head.csv,csv/rel8.csv \
--relationships=R_credit_2_account=header/rel9_head.csv,csv/rel9.csv \
--relationships=R_credit_2_company=header/rel10_head.csv,csv/rel10.csv \
--relationships=R_credit_2_device=header/rel11_head.csv,csv/rel11.csv \
--relationships=R_credit_2_ip=header/rel12_head.csv,csv/rel12.csv \
--relationships=R_credit_2_wifi=header/rel13_head.csv,csv/rel13.csv \
--relationships=R_operator_2_account=header/rel14_head.csv,csv/rel14.csv \
--relationships=R_withdraw_2_bankcard=header/rel15_head.csv,csv/rel15.csv \
--relationships=R_withdraw_2_device=header/rel16_head.csv,csv/rel16.csv \
--relationships=R_withdraw_2_ip=header/rel17_head.csv,csv/rel17.csv \
--relationships=R_withdraw_2_wifi=header/rel18_head.csv,csv/rel18.csv \
--delimiter "," \
--array-delimiter ";"
# --skip-bad-relationships \
#  --nodes person_0_0.csv --nodes place_0_0.csv \
#  --nodes post_0_0.csv --nodes tag_0_0.csv \
#  --nodes tagclass_0_0.csv 
#  --relationships comment_hasCreator_person_0_0.csv 
#  --relationships comment_hasTag_tag_0_0.csv 
#  --relationships comment_isLocatedIn_place_0_0.csv 
#  --relationships comment_replyOf_comment_0_0.csv 
#  --relationships comment_replyOf_post_0_0.csv 
#  --relationships forum_containerOf_post_0_0.csv 
#  --relationships forum_hasMember_person_0_0.csv 
#  --relationships forum_hasModerator_person_0_0.csv 
#  --relationships forum_hasTag_tag_0_0.csv 
#  --relationships organisation_isLocatedIn_place_0_0.csv 
#  --relationships person_hasInterest_tag_0_0.csv 
#  --relationships person_isLocatedIn_place_0_0.csv 
#  --relationships person_knows_person_0_0.csv 
#  --relationships person_likes_comment_0_0.csv 
