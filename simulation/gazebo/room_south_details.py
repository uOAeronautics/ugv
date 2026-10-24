"""South workshop objects inferred from the room rotation video (SI metres)."""
import math
SOUTH_COLORS={
 'detail_white':(.84,.85,.81),'detail_frame':(.34,.38,.40),
 'detail_dark':(.035,.042,.045),'detail_wood':(.66,.55,.36),
 'detail_blue':(.035,.18,.32),'detail_grey':(.47,.50,.51),
 'detail_card':(.46,.33,.20),'detail_tape':(.67,.55,.34),
}
def south_details():
    out=[]
    def group(prefix,x,y):
        def add(n,shape,p,size,c,rot=(0,0,0),collision=False):
            out.append(dict(name='d_south_'+prefix+'_'+n,shape=shape,pos=(x+p[0],y+p[1],p[2]),size=size,color='detail_'+c,rot=rot,collision=collision))
        def box(n,p,size,c,rot=(0,0,0),collision=True):add(n,'box',p,size,c,rot,collision)
        def cyl(n,p,r,h,c,rot=(0,0,0),collision=False):add(n,'cylinder',p,(r,h),c,rot,collision)
        def rod(n,a,b,r,c,collision=True):
            d=tuple(b[k]-a[k] for k in range(3));length=math.sqrt(sum(v*v for v in d))
            cyl(n,tuple((a[k]+b[k])/2 for k in range(3)),r,length,c,(0,math.acos(d[2]/length),math.atan2(d[1],d[0])),collision)
        return box,cyl,rod
    b,c,r=group('board',1.8,-6.2)
    b('surface',(0,0,1.15),(1.1,.045,1.30),'white')
    for sx in (-1,1):
        b('upright'+str(sx),(sx*.57,0,.98),(.035,.06,1.73),'frame')
        b('foot'+str(sx),(sx*.57,0,.10),(.06,.64,.045),'frame')
        for sy in (-1,1):c('wheel%s%s'%(sx,sy),(sx*.57,sy*.27,.055),.045,.04,'dark',(math.pi/2,0,0),True)
    for z in (.49,1.81):b('rail'+str(z),(0,0,z),(1.16,.06,.03),'frame')
    b('pen_tray',(0,.065,.49),(1.06,.12,.025),'frame')
    b('eraser',(.24,.07,.52),(.14,.055,.03),'dark',collision=False)
    # Irregular writing rows and clipped paper visible on the reference board.
    for j in range(22):
        b('writing'+str(j),(-.42+(j%3)*.29,.025,1.65-(j//3)*.14),(.16+.025*(j%2),.003,.005),'dark',collision=False)
    b('top_clip',(.30,.035,1.82),(.24,.012,.045),'card',collision=False)
    b('clipped_paper',(.31,.04,1.74),(.22,.002,.17),'white',collision=False)
    b,c,r=group('drawer_bench',4,-5)
    b('carcass',(0,0,.45),(1.3,1.8,.84),'frame')
    b('worktop',(0,0,.905),(1.38,1.87,.065),'wood')
    for k in range(7):
        z=.13+k*.105
        b('drawer'+str(k),(-.66,0,z),(.025,1.68,.092),'grey')
        b('handle'+str(k),(-.69,0,z+.015),(.035,.80,.018),'dark',collision=False)
    for sy in (-1,1):
        b('gantry_post'+str(sy),(.55,sy*.80,1.37),(.045,.045,.91),'frame')
    b('gantry_beam',(.55,0,1.82),(.045,1.65,.045),'frame')
    b('gantry_shelf',(.38,0,1.78),(.40,1.70,.035),'wood')
    for k,(xx,yy,ww,dd,hh) in enumerate(((-.22,-.45,.56,.49,.35),(.16,.28,.62,.68,.42),(.25,-.58,.25,.30,.18))):
        z=.94+hh/2
        b('box'+str(k),(xx,yy,z),(ww,dd,hh),'card')
        b('tape'+str(k),(xx,yy,z+hh/2+.002),(.055,dd,.004),'tape',collision=False)
    b,c,r=group('folded',0,-10.8)
    for j in range(3):
        xx=-.25+j*.18
        b('table'+str(j),(xx,0,.93),(.055,1.75,.78),'white',(0,.09,0))
        for sy in (-1,1):
            r('table_leg%s%s'%(j,sy),(xx-.05,sy*.65,.58),(xx-.05,sy*.65,1.26),.018,'frame')
    for j in range(3):
        xx=-1.6+j*.43
        for sx in (-1,1):r('frame_leg%s%s'%(j,sx),(xx+sx*.18,.50,.04),(xx+sx*.18,.50,1.36),.016,'dark')
        r('frame_top'+str(j),(xx-.18,.5,1.36),(xx+.18,.5,1.36),.016,'dark')
        r('frame_base'+str(j),(xx-.18,.5,.12),(xx+.18,.5,.12),.016,'dark')
    for j in range(2):
        xx=1+j*.53
        b('chairseat'+str(j),(xx,0,.45),(.42,.41,.035),'white')
        b('chairback'+str(j),(xx,-.19,.69),(.41,.045,.43),'white',(-.09,0,0))
        for sx in (-1,1):
            for sy in (-1,1):r('chairleg%s%s%s'%(j,sx,sy),(xx+sx*.17,sy*.15,.44),(xx+sx*.21,sy*.20,.025),.015,'white')
    b,c,r=group('rack',-5,-10.5)
    for sx in (-1,1):
        for sy in (-1,1):b('post%s%s'%(sx,sy),(sx*.63,sy*.28,.97),(.035,.035,1.94),'dark')
    for j in range(4):
        z=.15+j*.50
        b('shelf'+str(j),(0,0,z),(1.3,.60,.035),'dark')
        for k in range(2):
            xx=-.32+k*.64
            b('bin%d_%d'%(j,k),(xx,0,z+.16),(.51,.48,.28),'dark')
            b('label%d_%d'%(j,k),(xx,.246,z+.16),(.18,.005,.07),'white',collision=False)
            b('lid%d_%d'%(j,k),(xx,0,z+.31),(.54,.50,.025),'grey',collision=False)
            b('latch%d_%d'%(j,k),(xx+.18,.252,z+.27),(.055,.012,.055),'dark',collision=False)
    b,c,r=group('tray_stack',-3.95,-10.5)
    for j in range(12):
        z=.07+j*.085
        b('base'+str(j),(0,0,z),(.55,.66,.018),'grey')
        for sx in (-1,1):b('side%s%s'%(j,sx),(sx*.265,0,z+.04),(.02,.66,.075),'grey')
        for sy in (-1,1):b('end%s%s'%(j,sy),(0,sy*.32,z+.04),(.53,.02,.075),'grey')
    b,c,r=group('trash',-3,-9)
    c('body',(0,0,.29),.22,.56,'dark',collision=True)
    c('rim',(0,0,.58),.235,.045,'grey')
    c('inner',(0,0,.605),.197,.006,'dark')
    # The drawer island is a short cabinet, not a full-length workshop table.
    for part in out:
        if part['name'].startswith('d_south_drawer_bench_'):
            x,y,z=part['pos'];part['pos']=(x,-5+(y+5)*.72,z)
            if part['shape']=='box':
                a,b,c=part['size'];part['size']=(a,b*.72,c)
    return out
