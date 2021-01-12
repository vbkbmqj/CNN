ls ./check/1/*.t7 | xargs -I {} th test.lua -load {} |  tee test1.txt

--ls ./check/2/*.t7 | xargs -I {} th test.lua -load {} |  tee test2.txt
