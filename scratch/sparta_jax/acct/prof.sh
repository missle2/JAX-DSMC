#!/bin/bash
# usage: prof.sh TAG [profile.py args]   -- per-stage profile on a free physical core
tag=$1; shift
cd /home/aakash/jax_dsmc/JAX-DSMC-sparta
/home/aakash/jax_dsmc/run_locked.sh cpu bash -c "c=\$(scratch/sparta_jax/acct/freecore.sh); echo core \$c; rm -rf /tmp/sj_hlo_$tag; XLA_FLAGS='--xla_cpu_multi_thread_eigen=false --xla_dump_to=/tmp/sj_hlo_$tag --xla_dump_hlo_as_text --xla_dump_hlo_module_re=.*run.*' taskset -c \$c python3 ${PROFILE:-examples/sparta_jax/profile.py} --tag $tag --dump /tmp/sj_hlo_$tag $*" > /tmp/prof_$tag.txt 2>&1
grep "^core" /tmp/prof_$tag.txt
python3 scratch/sparta_jax/acct/show.py $tag || tail -20 /tmp/prof_$tag.txt
