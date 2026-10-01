#!/bin/bash
# print the even cpu id of the least busy physical core (both hyperthreads) over 2 s
head -17 /proc/stat > /tmp/fc1; sleep 2; head -17 /proc/stat > /tmp/fc2
paste /tmp/fc1 /tmp/fc2 | awk 'NR>1{u=($13+$14+$15)-($2+$3+$4); t=u+($16-$5); id=substr($1,4); busy[id]=(t>0? u/t : 0)} END{best=-1; for(c=14;c>=0;c-=2){b=busy[c]+busy[c+1]; if(best<0||b<bb){best=c;bb=b}} print best; printf "busy %.2f\n", bb > "/dev/stderr"}'
