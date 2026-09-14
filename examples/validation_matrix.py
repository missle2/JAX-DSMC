"""Write independent, reviewable validation jobs without running an ensemble.

Each configuration gets a pilot and a disjoint confirmatory seed range. Resolve
its baseline from THAT pilot before running confirmation; never reuse another
mesh/horizon's pilot implicitly. Resource and signal limits do not count as passes.
"""
import argparse
import json
from pathlib import Path


def jobs(seeds=2048):
    cases=[]
    parameters={'couette':['speed','temperature_left','temperature_right','accommodation','diameter','omega','alpha'],
                'shock':['temp','ux','diam','omega','alpha'],
                'inlet-shock':['temp','nrho','mach','diam','omega','alpha']}
    for flow,names in parameters.items():
        baseline=dict(particles=2048,cells=16,steps=20,dt=.05,epsilon=.005 if flow=='couette' else 0.,attempts=4096)
        levels=[('base',baseline)]
        for factor in (2,4):
            levels.extend([
                (f'dt_{factor}',{**baseline,'dt':baseline['dt']/factor,'steps':baseline['steps']*factor,'epsilon':baseline['epsilon']/factor}),
                (f'mesh_{factor}',{**baseline,'cells':baseline['cells']*factor,'particles':baseline['particles']*factor}),
                (f'population_{factor}',{**baseline,'particles':baseline['particles']*factor}),
                (f'horizon_{factor}',{**baseline,'steps':baseline['steps']*factor})])
        if flow=='couette':
            levels.extend((f'epsilon_{factor}',{**baseline,'epsilon':baseline['epsilon']/factor}) for factor in (2,4))
        for name in names:
            for level,cfg in levels:
                identifier=f'{flow}_{name}_{level}'
                stem=['python','-m','examples.gradient_convergence','--flow',flow,'--parameter',name]
                for key,value in cfg.items():
                    stem.extend(['--'+key.replace('_','-'),str(value)])
                pilot=stem+['--seeds','32','--output',f'results/{identifier}_pilot.json']
                confirm=stem+['--seeds',str(seeds),'--seed-start','32','--baseline','PILOT_OBJECTIVE_MEAN',
                              '--output',f'results/{identifier}_confirm.json']
                cases.append(dict(id=identifier,axis=level,config=cfg,pilot_argv=pilot,
                                  confirmation_argv=confirm,baseline_source=f'results/{identifier}_pilot.json',
                                  acceptance='Requires resolved FD plateau and paired refinement equivalence; not seed-only agreement'))
    base=dict(particles=2000,cells=40,steps=20,retained=128,source_samples=512,partners=8,neighbors=32)
    levels=[("base",base)]
    for axis in ("particles","retained","source_samples","partners","neighbors"):
        for factor in (2,4):
            levels.append((f"{axis}_{factor}",{**base,axis:base[axis]*factor}))
    for reconstruction in ("affine","ballistic"):
        for level,cfg in levels:
            identifier=f"p3_inlet_{reconstruction}_{level}"
            command=["python","-m","examples.p3_replication","--backend","jax","--reconstruction",reconstruction,"--seeds",str(seeds)]
            for key,value in cfg.items():
                command.extend(["--"+key.replace("_","-"),str(value)])
            command.extend(["--output",f"results/{identifier}.json"])
            cases.append(dict(id=identifier,axis=level,config=cfg,argv=command,
                              acceptance="P3 kinetic/reconstruction convergence, not a finite-sample VJP identity"))
    return cases


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',type=Path,default=Path('validation_jobs.json'))
    parser.add_argument('--seeds',type=int,default=2048)
    args=parser.parse_args()
    result={'jobs':jobs(args.seeds),'execution':'Not launched; replace PILOT_OBJECTIVE_MEAN from each independent pilot.',
            'environment':'source /home/aakash/jax_dsmc/activate_jax_env.sh',
            'limitations':['Transient horizon sweep is not stationary sensitivity validation.',
                           'Inlet count means at integer support require one-sided replay.',
                           'Capacity failures require increasing attempts/storage and restarting invalid trials.',
                           'P3 reconstruction convergence is a separate family from NTC flow validation.']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(f'Wrote {len(result["jobs"])} unlaunched jobs to {args.output}')


if __name__=='__main__':
    main()
