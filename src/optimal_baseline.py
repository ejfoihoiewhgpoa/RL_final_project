import sys, copy, json
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collections import deque
from rescue_mission import RescueEnv, load_maps

def setst(e,s):
    d=e.decode(s); e.pos=d['pos']; e.mask=d['mask']; e.carry=d['carry']; e.med=d['med']; e.supply_used=d['supply_used']; e.t=0

def reachable(e):
    e.reset(); s0=e._encode(); seen={s0}; q=deque([s0]); trans={}
    while q:
        s=q.popleft()
        for a in range(6):
            setst(e,s); e.max_steps=10**9
            s2,r,te,tr,_=e.step(a); trans[(s,a)]=(s2,r,te)
            if s2 not in seen: seen.add(s2); q.append(s2)
    return s0,seen,trans

def solve(e,gamma,iters=2000):
    s0,seen,trans=reachable(e)
    V={s:0.0 for s in seen}
    for _ in range(iters):
        d=0
        for s in seen:
            best=max(trans[(s,a)][1]+(0 if trans[(s,a)][2] else gamma*V[trans[(s,a)][0]]) for a in range(6))
            d=max(d,abs(best-V[s])); V[s]=best
        if d<1e-6: break
    # rollout
    s=s0; order=[]; G=0; steps=0; haz=0
    while steps<400:
        a=max(range(6),key=lambda a: trans[(s,a)][1]+(0 if trans[(s,a)][2] else gamma*V[trans[(s,a)][0]]))
        s2,r,te=trans[(s,a)]
        G+=r; steps+=1
        if e.decode(s2)['mask']!=e.decode(s)['mask']:
            m=e.decode(s2)['mask']^e.decode(s)['mask']; order.append(m.bit_length()-1)
        if e.decode(s2)['pos'] in e.hazards: haz+=1
        s=s2
        if te: break
    return order,steps,G,haz

if __name__=="__main__":
    rew=json.loads(sys.argv[1]) if len(sys.argv)>1 else None
    gamma=float(sys.argv[2]) if len(sys.argv)>2 else 0.99
    for name in load_maps():
        for mode in "AB":
            e=RescueEnv(name,reward_mode=mode,reward_cfg=rew)
            print(name,mode,solve(e,gamma))
