#!/bin/bash
# usage: prof.sh TAG [profile.py args]
tag=$1; shift
cd /home/aakash/jax_dsmc/JAX-DSMC-sparta
rm -rf /tmp/sj_hlo
/home/aakash/jax_dsmc/run_locked.sh cpu env XLA_FLAGS="--xla_cpu_multi_thread_eigen=false --xla_dump_to=/tmp/sj_hlo --xla_dump_hlo_as_text --xla_dump_hlo_module_re=.*run.*" taskset -c 3 python3 examples/sparta_jax/profile.py --tag $tag "$@" > /tmp/prof_$tag.txt 2>&1
python3 -c "
import json; d=json.load(open('examples/sparta_jax/results/profile_$tag.json'))
print('wall', round(d['wall_ms_per_step'],3), 'ops', round(d['op_ms_per_step'],3), d['ops_per_step'])
for k,v in d['stages'].items(): print(' ', k, round(v['ms_per_step'],3), v['ops_per_step'])
for s,o,ms in d['top_ops'][:14]: print('   ', s,o,round(ms,3))
" || tail -20 /tmp/prof_$tag.txt
