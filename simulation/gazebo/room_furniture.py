"""Photo-informed workshop furniture. SI dimensions, estimated from room video.
Reusable flat parts for Gazebo and Blender. Static collision on structural parts.
"""
import math
import os
import tempfile
from pathlib import Path
FURNITURE_COLORS={
 'room_red':(.63,.012,.027),'room_red_edge':(.36,.008,.015),
 'room_black':(.018,.021,.025),'room_rubber':(.026,.029,.034),
 'room_grey':(.27,.30,.32),'room_metal':(.48,.52,.54),
 'room_wood':(.72,.61,.40),'room_wood_edge':(.52,.42,.27),
 'room_cardboard':(.43,.30,.17),'room_tape':(.67,.54,.32),
 'room_white':(.86,.87,.82),'room_screen':(.025,.055,.065),
 'room_blue':(.018,.17,.32),'room_green':(.03,.23,.11),
 'room_orange':(.86,.24,.025),'room_cork':(.43,.31,.19),
}
HERE=Path(__file__).resolve().parent

def _atomic_text(path, text):
    """Publish complete mesh bytes without racing concurrent scene readers."""
    if path.exists() and path.read_text() == text:
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.' + path.name + '.', suffix='.tmp',
                                         delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _chair_shell():
    """Moulded polypropylene seat/back with curved edges and lumbar sweep."""
    path=HERE/'assets/room/chair_shell.obj';path.parent.mkdir(parents=True,exist_ok=True)
    verts=[];faces=[]
    # Longitudinal seat-to-back profile, rear is local +Y.
    profile=[(-.235,.445),(-.20,.436),(-.10,.426),(.02,.426),(.12,.44),(.18,.49),(.205,.58),(.22,.68),(.24,.79),(.25,.83)]
    for layer in (-.005,.005):
        for j,(y,z) in enumerate(profile):
            half=.225 if j<7 else .218-(j-7)*.008
            for k in range(9):
                u=-1+k/4
                verts.append((u*half,y,z+.025*u*u+layer))
    n=len(profile)*9
    for layer in range(2):
        off=layer*n
        for j in range(len(profile)-1):
            for k in range(8):
                a=off+j*9+k;b=a+1;c=a+10;d=a+9
                faces.extend(((a,b,c),(a,c,d)) if layer else ((a,c,b),(a,d,c)))
    for j in range(len(profile)-1):
        for k in (0,8):
            a=j*9+k;b=(j+1)*9+k
            faces.extend(((a,b,b+n),(a,b+n,a+n)))
    for j in (0,len(profile)-1):
        for k in range(8):
            a=j*9+k;b=a+1;faces.extend(((a,b,b+n),(a,b+n,a+n)))
    _atomic_text(path, '# Curved moulded sled chair shell\n'+''.join('v %.6f %.6f %.6f\n'%v for v in verts)+''.join('f %d %d %d\n'%tuple(q+1 for q in f) for f in faces))
    return 'assets/room/chair_shell.obj'

def furniture_assembly():
    parts=[];shell=_chair_shell()
    def group(prefix,x,y,yaw=0):
        def add(n,shape,pos,size,color,rot=(0,0,0),collision=False,**extra):
            xx,yy,z=pos;c=math.cos(yaw);s=math.sin(yaw)
            parts.append(dict(name=prefix+'_'+n,shape=shape,pos=(x+c*xx-s*yy,y+s*xx+c*yy,z),size=size,color='room_'+color,rot=(rot[0],rot[1],rot[2]+yaw),collision=collision,**extra))
        def box(n,p,s,c,rot=(0,0,0),collision=True):add(n,'box',p,s,c,rot,collision)
        def cyl(n,p,r,h,c,rot=(0,0,0),collision=False):add(n,'cylinder',p,(r,h),c,rot,collision)
        def rod(n,a,b,r,c,collision=True):
            d=[b[i]-a[i] for i in range(3)];h=math.sqrt(sum(v*v for v in d))
            cyl(n,tuple((a[i]+b[i])/2 for i in range(3)),r,h,c,(0,math.acos(d[2]/h),math.atan2(d[1],d[0])),collision)
        return add,box,cyl,rod
    def table(n,x,y,yaw=0,w=1.6,d=.8):
        a,b,c,r=group(n,x,y,yaw)
        b('laminate',(0,0,.742),(w,d,.036),'wood');b('edge',(0,0,.72),(w+.004,d+.004,.012),'wood_edge')
        for s in (-1,1):
            xx=s*(w/2-.18)
            b('upright'+str(s),(xx,0,.385),(.075,.075,.64),'metal')
            b('foot'+str(s),(xx,0,.055),(.065,d-.10,.045),'metal')
            for k in (-1,1):b('endpad%s%s'%(s,k),(xx,k*(d/2-.08),.026),(.085,.08,.027),'rubber')
        b('crossbar',(0,0,.51),(w-.30,.045,.08),'grey')
        return a,b,c,r
    def chair(n,x,y,yaw=0,black=False):
        a,b,c,r=group(n,x,y,yaw)
        a('shell','mesh',(0,0,0),(1,1,1),'black' if black else 'red',mesh=shell)
        b('seat_collision',(0,0,.44),(.43,.40,.025),'red' if not black else 'black',collision=True)
        for s in (-1,1):
            r('front'+str(s),(s*.205,-.18,.43),(s*.23,-.25,.025),.011,'black')
            r('rear'+str(s),(s*.205,.16,.48),(s*.23,.29,.025),.011,'black')
            r('sled'+str(s),(s*.23,-.25,.025),(s*.23,.29,.025),.011,'black')
    def office(n,x,y,yaw=0):
        a,b,c,r=group(n,x,y,yaw)
        c('column',(0,0,.29),.032,.37,'metal',collision=True)
        c('collar',(0,0,.17),.049,.12,'black')
        for k in range(5):
            t=k*math.tau/5;ex=.29*math.cos(t);ey=.29*math.sin(t)
            r('spoke'+str(k),(0,0,.13),(ex,ey,.075),.019,'black')
            c('caster'+str(k),(ex,ey,.045),.043,.05,'rubber',(math.pi/2,0,t))
        b('seat',(0,0,.47),(.49,.47,.075),'black')
        b('back',(0,.22,.72),(.46,.07,.43),'black',(.12,0,0))
        for s in (-1,1):
            r('armstem'+str(s),(s*.23,.11,.49),(s*.29,.11,.65),.013,'grey')
            b('armpad'+str(s),(s*.29,.015,.65),(.06,.30,.04),'black')
    def workstation(n,x,y,yaw=0):
        a,b,c,r=table(n,x,y,yaw,w=1.65,d=.75)
        for j,xx in enumerate((-.37,.25)):
            b('monitor_foot'+str(j),(xx,.16,.777),(.25,.18,.018),'black')
            b('monitor_stand'+str(j),(xx,.19,.90),(.04,.04,.24),'grey')
            b('monitor'+str(j),(xx,.205,1.09),(.55,.045,.33),'black')
            b('display'+str(j),(xx,.179,1.09),(.505,.004,.285),'screen',collision=False)
        b('keyboard',(0,-.17,.775),(.44,.15,.02),'black')
        for row in range(4):
            for k in range(12):b('key%d_%d'%(row,k),(-.195+k*.035,-.215+row*.029,.788),(.027,.021,.006),'grey',collision=False)
        b('mouse',(.36,-.16,.783),(.05,.085,.025),'black')
        b('tower',(.62,.12,.30),(.22,.42,.54),'black')
        for k in range(5):b('tower_vent'+str(k),(.62,-.095,.35+k*.022),(.15,.005,.006),'grey',collision=False)
        cs=math.cos(yaw);sn=math.sin(yaw);office(n+'_chair',x+sn*.7,y-cs*.7,yaw)
    def carton(b,n,p,size):
        b(n,p,size,'cardboard');b(n+'_tape',(p[0],p[1],p[2]+size[2]/2+.001),(.05,size[1],.002),'tape',collision=False)
    def toolchest(n,x,y,yaw=0,w=.9,h=1.05):
        a,b,c,r=group(n,x,y,yaw)
        b('body',(0,0,h/2+.08),(w,.50,h),'red');b('top',(0,0,h+.09),(w+.025,.53,.025),'black')
        for j in range(7):
            z=.19+j*(h-.12)/7
            b('drawer'+str(j),(0,-.256,z),(w-.05,.018,.10),'red_edge')
            b('pull'+str(j),(0,-.277,z+.025),(w-.15,.025,.018),'metal',collision=False)
        for sx in (-1,1):
            for sy in (-1,1):c('wheel%s%s'%(sx,sy),(sx*(w/2-.09),sy*.19,.065),.06,.045,'rubber',(math.pi/2,0,0),True)
    def shelf(n,x,y,yaw=0):
        a,b,c,r=group(n,x,y,yaw)
        for sx in (-1,1):
            for sy in (-1,1):b('post%s%s'%(sx,sy),(sx*.72,sy*.24,1.05),(.04,.04,2.1),'grey')
        for j in range(5):
            z=.12+j*.46;b('shelf'+str(j),(0,0,z),(1.5,.55,.045),'wood')
            for k in range(3):
                xx=-.48+k*.47
                if (j+k)%2:carton(b,'carton%d_%d'%(j,k),(xx,0,z+.16),(.38,.40,.28))
                else:
                    b('bin%d_%d'%(j,k),(xx,0,z+.13),(.38,.41,.22),'blue')
                    b('label%d_%d'%(j,k),(xx,-.207,z+.13),(.15,.003,.06),'white',collision=False)
    # Perimeter arrangement leaves a generous central manoeuvring area.
    for i,(x,y) in enumerate(((2.0,3.5),(3.65,3.5),(2.0,4.35),(3.65,4.35),(-8.3,.5),(-8.3,1.4))):
        a,b,c,r=table('teaching_table'+str(i),x,y)
        if i in (0,2,4):
            b('notebook',(-.3,.07,.773),(.28,.21,.015),'white',collision=False)
            b('electronics',(.3,-.08,.79),(.21,.13,.043),'green',collision=False)
            for k in range(4):b('chip'+str(k),(.24+k*.035,-.08,.815),(.022,.04,.009),'black',collision=False)
        if i < 2:
            chair('chair_a'+str(i),x-.4,y-.75,0)
        elif i < 4:
            chair('chair_b'+str(i),x+.4,y+.75,math.pi)
        else:
            chair('chair_a'+str(i),x-.4,y-.75,0,black=i==4)
    # Outer end chairs around the joined teaching tables, never between tops.
    for j,y in enumerate((3.5,4.35)):
        chair('teaching_end_left'+str(j),.45,y,-math.pi/2)
        chair('teaching_end_right'+str(j),5.15,y,math.pi/2)
    for i,x in enumerate((6.5,8.2,9.9)):workstation('computer'+str(i),x,5.15)
    for i,(x,y,yaw) in enumerate(((-6,-3,0),(-7.15,-3,0),(5,-4.8,0))):toolchest('tools'+str(i),x,y,yaw)
    shelf('supply_shelf',10.9,-4.9);shelf('rear_shelf',-9.9,-4.9)
    # Shop island with grey welded frame and broad wood work surface.
    a,b,c,r=table('shop_island',7,-3,w=3.4,d=1.25)
    b('lower_shelf',(0,0,.24),(3.15,1.1,.05),'grey')
    for i,xx in enumerate((-.95,-.20,.7)):carton(b,'box'+str(i),(xx,.1,.91),(.55,.45,.28))
    b('bench_vise',(1.25,-.43,.87),(.22,.17,.19),'blue');b('vise_jaws',(1.25,-.43,.975),(.26,.13,.04),'metal')
    r('vise_handle',(1.25,-.56,.85),(1.25,-.56,1.06),.009,'metal',False)
    # Tall grey cupboards and black equipment lockers.
    for j,x in enumerate((-10.5,-9.55,-8.6)):
        a,b,c,r=group('locker'+str(j),x,5.48)
        b('body',(0,0,1.0),(.88,.65,2.0),'grey' if j<2 else 'black')
        for s in (-1,1):
            b('door'+str(s),(s*.214,-.334,1.02),(.416,.018,1.91),'metal' if j<2 else 'black')
            b('handle'+str(s),(s*.075,-.36,1.0),(.015,.025,.15),'black')
            for k in range(4):b('vent%s%s'%(s,k),(s*.214,-.346,1.75-k*.035),(.25,.005,.009),'grey',collision=False)
    # Cork partition with Canadian flag seen in the workshop panorama.
    a,b,c,r=group('flag_partition',6.8,-4.7,math.pi)
    b('panel',(0,0,1.20),(3.4,.10,1.5),'cork')
    for s in (-1,1):
        b('post'+str(s),(s*1.73,0,.95),(.09,.14,1.9),'grey')
        b('foot'+str(s),(s*1.73,0,.065),(.12,.75,.08),'grey')
    b('flag',(0,-.057,1.25),(1.35,.008,.675),'white',collision=False)
    for s in (-1,1):b('flag_red'+str(s),(s*.50,-.064,1.25),(.35,.008,.675),'red',collision=False)
    # Geometric maple silhouette assembled from rotated angular lobes.
    b('maple_core',(0,-.072,1.25),(.19,.005,.25),'red',collision=False)
    for s in (-1,1):
        b('maple_lobe'+str(s),(s*.12,-.072,1.28),(.14,.005,.19),'red',(0,s*.50,0),False)
        b('maple_outer'+str(s),(s*.19,-.072,1.24),(.13,.005,.10),'red',(0,s*.65,0),False)
    b('maple_top',(0,-.072,1.40),(.085,.005,.15),'red',collision=False)
    b('maple_stem',(0,-.072,1.08),(.02,.005,.12),'red',collision=False)
    # Storage tubs and buckets under the partition.
    for j in range(3):b('tub'+str(j),(-1+j*.6,0,.17),(.48,.42,.28),'blue' if j==0 else 'black')
    c('bucket',(.96,0,.20),.15,.32,'white');c('bucket_lid',(.96,0,.37),.16,.025,'orange')
    # Foam aircraft and unfinished foam stock stored on the workshop wall.
    a,b,c,r=group('foam_aircraft',-3.8,-5.65)
    b('fuselage',(0,0,2.0),(.10,.08,1.25),'white',collision=False)
    for side in (-1,1):
        b('wing'+str(side),(side*.34,0,2.02),(.72,.035,.27),'blue',(0,-side*.35,0),False)
        b('tail'+str(side),(side*.15,0,1.52),(.35,.035,.15),'blue',(0,-side*.25,0),False)
    b('cockpit',(0,-.06,2.22),(.08,.065,.23),'black',collision=False)
    a,b,c,r=group('white_foam_stock',-5.4,-5.42)
    for j in range(4):
        b('board'+str(j),(j*.09,0,.72),(.07,.54,1.45),'white',(0,.10+j*.025,.025*j))
    # Reference-view accessories: open laptops, power leads and workshop storage.
    for j,(xx,yy) in enumerate(((2.15,3.55),(3.85,3.48))):
        a,b,c,r=group('ref_laptop'+str(j),xx,yy,.06-j*.16)
        b('base',(0,0,.781),(.34,.24,.018),'metal')
        b('keyboard',(0,.015,.792),(.29,.12,.004),'black',collision=False)
        b('trackpad',(0,-.075,.792),(.09,.045,.004),'grey',collision=False)
        b('screen',(0,.123,.907),(.34,.014,.235),'black',(-.12,0,0))
        b('display',(0,.113,.907),(.311,.003,.205),'screen',(-.12,0,0),False)
        for k in range(5):
            b('screen_line'+str(k),(-.05,.109,.96-k*.022),(.12+k*.015,.002,.004),'metal',(-.12,0,0),False)
    a,b,c,r=group('ref_power',2.77,3.54)
    b('strip',(0,0,.785),(.07,.34,.035),'white')
    for k in range(4):
        b('socket'+str(k),(0,-.12+k*.078,.805),(.035,.045,.004),'grey',collision=False)
    cable=[(.02,.18,.79),(.13,.30,.78),(.30,.39,.78),(.42,.42,.76),(.48,.48,.65),(.49,.49,.06),(.38,.35,.018),(.25,.27,.018)]
    for k,(v,w) in enumerate(zip(cable,cable[1:])):r('lead'+str(k),v,w,.004,'black',False)
    a,b,c,r=group('ref_backpack',3.4,2.5,-.25)
    b('body',(0,0,.25),(.30,.19,.46),'black',(.12,0,0))
    b('pocket',(0,-.11,.20),(.25,.07,.25),'rubber',(.12,0,0))
    for side in (-1,1):
        r('zip'+str(side),(side*.13,-.15,.10),(side*.13,-.14,.33),.004,'grey',False)
        r('strap'+str(side),(side*.11,.105,.39),(side*.13,.19,.08),.013,'black',False)
    r('handle',(-.06,0,.49),(.06,0,.49),.013,'black',False)
    for j,(xx,yy) in enumerate(((-6.4,-2.9),(-7.8,-3.2))):
        a,b,c,r=group('ref_stool'+str(j),xx,yy)
        c('seat',(0,0,.53),.19,.07,'black',collision=True)
        c('post',(0,0,.31),.025,.40,'metal',collision=True)
        for k in range(5):
            t=k*math.tau/5;ex=.25*math.cos(t);ey=.25*math.sin(t)
            r('spoke'+str(k),(0,0,.12),(ex,ey,.07),.013,'grey')
            c('caster'+str(k),(ex,ey,.04),.035,.035,'black',(math.pi/2,0,t))
    a,b,c,r=group('ref_stacked_tables',10.5,-3.7,.08)
    for j in range(5):
        xx=(j-2)*.095
        b('top'+str(j),(xx,0,.95),(.055,1.58,.80),'wood',(0,.06,0))
        for sy in (-1,1):
            r('folded_leg%s%s'%(j,sy),(xx-.035,sy*.60,.57),(xx-.035,sy*.60,1.29),.021,'metal')
    for sy in (-1,1):b('rack_foot'+str(sy),(0,sy*.65,.06),(.90,.07,.07),'grey')
    for j in range(5):
        a,b,c,r=group('ref_stacked_chair'+str(j),8,-4.7+j*.045,.1)
        a('shell','mesh',(0,0,j*.075),(1,1,1),'black',mesh=shell)
        for side in (-1,1):
            r('front'+str(side),(side*.205,-.18,.43+j*.075),(side*.23,-.25,.025+j*.075),.011,'black')
            r('rear'+str(side),(side*.205,.16,.48+j*.075),(side*.23,.29,.025+j*.075),.011,'black')
            r('sled'+str(side),(side*.23,-.25,.025+j*.075),(side*.23,.29,.025+j*.075),.011,'black')
    return parts
