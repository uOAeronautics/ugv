"""Door-side meeting bay observed in the second walking reference (metres)."""
import math
WALK_COLORS={'walk_grey':(.30,.33,.34),'walk_dark':(.026,.03,.032),'walk_paper':(.81,.80,.73),'walk_red':(.65,.02,.025),'walk_blue':(.025,.18,.34),'walk_screen':(.22,.34,.44),'walk_wood':(.58,.43,.27)}
def walk_parts():
    p=[]
    def box(n,xyz,size,c='walk_dark',rot=(0,0,0),collision=False):p.append(dict(name='walk_'+n,shape='box',pos=xyz,size=size,color=c,rot=rot,collision=collision))
    def rod(n,a,b,r=.006,c='walk_dark'):
        d=[b[i]-a[i] for i in range(3)];h=math.sqrt(sum(v*v for v in d))
        p.append(dict(name='walk_'+n,shape='cylinder',pos=tuple((a[i]+b[i])/2 for i in range(3)),size=(r,h),color=c,rot=(0,math.acos(d[2]/h),math.atan2(d[1],d[0])),collision=False))
    # Open laptop and grey prototype on the joined meeting tables.
    box('laptop_base',(-3.45,-3.20,.775),(.34,.25,.018),'walk_grey')
    box('laptop_lid',(-3.45,-3.08,.90),(.34,.018,.25),'walk_dark',(-.14,0,0))
    box('laptop_display',(-3.45,-3.093,.90),(.31,.004,.22),'walk_screen',(-.14,0,0))
    box('folded_prototype',(-3.4,-4.18,.84),(.46,.32,.15),'walk_grey',(0,.08,.15))
    for j,(x,y) in enumerate(((-3.1,-4.5),(-3.7,-3.7),(-4.0,-4.3))):
        box('paper'+str(j),(x,y,.771),(.29,.21,.003),'walk_paper',(0,0,j*.21))
    box('red_sheet',(-3.0,-3.8,.774),(.30,.23,.003),'walk_red')
    box('radio',(-2.94,-4.7,.80),(.19,.24,.065),'walk_blue')
    for k in (-1,1):
        box('radio_stick'+str(k),(-2.94+k*.055,-4.7,.845),(.016,.016,.04))
    rod('radio_antenna',(-2.94,-4.8,.82),(-2.94,-4.96,1.00),.004)
    for j,(x,y) in enumerate(((-3.0,-3.4),(-3.1,-4.2),(-4.05,-3.65))):
        for k in range(16):
            a=k*math.tau/16;b=(k+1)*math.tau/16
            rod(f'lead_{j}_{k}',(x+.06*math.cos(a),y+.085*math.sin(a),.782),(x+.06*math.cos(b),y+.085*math.sin(b),.782),.0025)
    # Tall folded rolling grid frames immediately beside the south-west door.
    for j in range(2):
        x=-5.30+j*.12;y=-9.9
        for sy in (-1,1):rod(f'rack_post{j}_{sy}',(x,y+sy*.48,.10),(x,y+sy*.48,1.77),.018)
        for k in range(10):rod(f'rack_cross{j}_{k}',(x,y-.48,.16+k*.17),(x,y+.48,.16+k*.17),.008,'walk_grey')
        for k in range(5):rod(f'rack_grid{j}_{k}',(x,y-.43+k*.215,.16),(x,y-.43+k*.215,1.70),.005,'walk_grey')
        for sy in (-1,1):box(f'rack_foot{j}_{sy}',(x,y+sy*.48,.08),(.55,.045,.04))
    box('leaning_door_board',(-5.16,-10.6,.73),(.06,.42,1.43),'walk_wood',(0,-.10,0),True)
    return p
