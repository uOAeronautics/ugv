"""West window bays refined against reference seconds 11,13,15.
Cardinal SI frame E+X,N+Y. Replace generic northwest shop/airframe pieces.
"""
import math
BAY_COLORS={
 'bay_black':(.015,.019,.022),'bay_steel':(.16,.18,.19),
 'bay_silver':(.56,.59,.60),'bay_white':(.78,.79,.75),
 'bay_red':(.61,.02,.025),'bay_wood':(.55,.39,.22),
 'bay_pale':(.72,.63,.44),'bay_blue':(.025,.12,.25),
 'bay_lime':(.56,.72,.035),'bay_cyan':(.015,.28,.49),'bay_paper':(.67,.67,.61),'bay_wire':(.055,.065,.07),
}

def bay_parts():
    parts=[]
    def box(n,p,s,c='bay_black',r=(0,0,0),collision=False):
        parts.append(dict(name='westbay_'+n,shape='box',pos=p,size=s,color=c,rot=r,collision=collision))
    def cyl(n,p,rad,h,c='bay_black',r=(0,0,0),collision=False):
        parts.append(dict(name='westbay_'+n,shape='cylinder',pos=p,size=(rad,h),color=c,rot=r,collision=collision))
    def rod(n,a,b,rad=.01,c='bay_silver',collision=False):
        d=[b[i]-a[i] for i in range(3)];h=math.sqrt(sum(q*q for q in d))
        cyl(n,tuple((a[i]+b[i])/2 for i in range(3)),rad,h,c,(0,math.acos(d[2]/h),math.atan2(d[1],d[0])),collision)
    def ring(n,x,y,z,rad,tube,c='bay_silver',plane='xy'):
        for j in range(20):
            a=j*math.tau/20;b=(j+1)*math.tau/20
            def point(v):
                return (x+rad*math.cos(v),y+rad*math.sin(v),z) if plane=='xy' else (x,y+rad*math.cos(v),z+rad*math.sin(v))
            rod(n+str(j),point(a),point(b),tube,c)
    # A coherent inward-projecting steel work island; the panorama sees its
    # broad front at13s and its narrow end plus foam-stock at15s.
    x,y=-3.8,4.4
    box('shop_top',(x,y,.90),(1.10,2.20,.055),'bay_pale',collision=True)
    for sx in (-1,1):
        for sy in (-1,1):
            xx=x+sx*.47;yy=y+sy*1.01
            box(f'shop_leg_{sx}_{sy}',(xx,yy,.45),(.055,.055,.88),'bay_steel',collision=True)
    for sx in (-1,1):box('shop_side_rail'+str(sx),(x+sx*.47,y,.26),(.045,2.03,.045),'bay_steel',collision=True)
    box('shop_lower_shelf',(x,y,.25),(.94,1.96,.035),'bay_black',collision=True)
    box('shop_under_case',(x-.12,y+.22,.43),(.55,.7,.31),'bay_black',collision=True)
    # Aluminium/foam airframe lying on the top. Slender fuselage, open wing
    # spars/ribs and a white skin panel distinguish it from a generic cross.
    rod('airframe_fuselage',(x+.08,y-.88,.98),(x+.08,y+.83,1.01),.024,'bay_silver')
    for yy in (y-.16,y+.07):rod('airframe_spar_'+str(yy),(x-.69,yy,1.01),(x+.69,yy,1.01),.012,'bay_silver')
    for j in range(9):
        xx=x-.65+j*.16;rod('airframe_wing_rib_'+str(j),(xx,y-.18,1.008),(xx,y+.14,1.015),.008,'bay_silver')
    box('airframe_skin',(x-.27,y-.02,1.029),(.56,.33,.012),'bay_white')
    box('airframe_tail',(x+.08,y+.68,1.04),(.46,.16,.013),'bay_white')
    box('airframe_fin',(x+.08,y+.74,1.12),(.018,.18,.18),'bay_red',(0,.08,0))
    box('airframe_redstripe',(x-.26,y-.20,1.04),(.57,.045,.009),'bay_red')
    # Second angular loose airframe/wing sections behind the main plane.
    box('spare_wing_left',(x-.24,y+.57,.99),(.43,.44,.025),'bay_white',(0,.08,-.32))
    box('spare_wing_red',(x-.30,y+.57,1.008),(.32,.06,.012),'bay_red',(0,.08,-.32))
    # Backstop project display, white lower field and broad charcoal title band.
    box('display_board',(-4.15,5.62,1.47),(1.60,.045,.93),'bay_white',collision=True)
    box('display_header',(-4.15,5.59,1.83),(1.60,.009,.15),'bay_steel')
    for j in range(8):box('display_title_stroke'+str(j),(-4.65+j*.13,5.583,1.83),(.085,.002,.017),'bay_white')
    for j in range(6):box('display_note_'+str(j),(-4.55+(j%3)*.38,5.586,1.55-(j//3)*.16),(.27,.002,.012),'bay_black')
    # Tall upright red/white foam aircraft parts between pier and project board.
    for j in range(3):
        xx=-4.95+j*.15
        box('vertical_foam_'+str(j),(xx,5.42,1.62),(.10,.055,1.18),'bay_white',(0,.035*j,0))
        box('vertical_red_'+str(j),(xx,5.385,1.91),(.095,.006,.38),'bay_red')
    box('upright_tailplane',(-4.8,5.43,1.62),(.60,.055,.18),'bay_red')
    # Three individual long foam boards leaning against the front-right end.
    for j in range(3):
        box('leaning_foam_'+str(j),(-2.93+j*.035,3.94+j*.13,.74),(.075,.42,1.47),'bay_white',(0,-.12,.035*j),True)
    # Separate small RC aircraft on narrow silver T support in front of bench.
    xx,yy=-3.35,3.13
    rod('rc_stand_upright',(xx,yy,.05),(xx,yy,.78),.013,'bay_silver',True)
    rod('rc_stand_foot',(xx-.30,yy,.04),(xx+.30,yy,.04),.016,'bay_silver',True)
    rod('rc_stand_foot2',(xx,yy-.20,.04),(xx,yy+.20,.04),.014,'bay_silver',True)
    box('rc_fuselage',(xx,yy,.83),(.83,.06,.055),'bay_white',(0,.035,-.12))
    box('rc_wing',(xx-.05,yy,.86),(.25,.92,.016),'bay_white',(0,0,-.12))
    box('rc_wing_red',(xx-.05,yy+.26,.872),(.25,.27,.006),'bay_red',(0,0,-.12))
    box('rc_tail',(xx+.33,yy,.87),(.11,.34,.015),'bay_red')
    box('rc_vertical_tail',(xx+.35,yy,.94),(.13,.018,.14),'bay_white')
    cyl('rc_nose',(xx-.43,yy,.83),.03,.07,'bay_blue',(0,math.pi/2,0))
    # Black backed shop stool on chrome five-leg base with circular foot ring.
    xx,yy=-4.55,2.95
    cyl('shop_stool_seat',(xx,yy,.56),.185,.07,'bay_black',collision=True)
    rod('shop_stool_post',(xx,yy,.09),(xx,yy,.53),.023,'bay_silver',True)
    ring('shop_stool_ring_',xx,yy,.25,.21,.010)
    box('shop_stool_back',(xx,yy+.14,.79),(.29,.04,.23),'bay_black',(0,.05,0))
    for sx in (-1,1):rod('shop_stool_back_post'+str(sx),(xx+sx*.10,yy+.13,.55),(xx+sx*.10,yy+.14,.79),.012,'bay_silver')
    for j in range(5):
        a=j*math.tau/5;end=(xx+.25*math.cos(a),yy+.25*math.sin(a),.055)
        rod('shop_stool_foot'+str(j),(xx,yy,.10),end,.014,'bay_silver')
        cyl('shop_stool_caster'+str(j),end,.03,.035,'bay_black',(math.pi/2,0,a))
    # Add realistic contents to the existing west folding instrument table.
    # Its open black case remains supplied by room_details.py.
    for j in range(4):
        box('instrument_box_'+str(j),(-5.48+j*.25,1.82,.91),(.18,.14,.07),'bay_white')
        cyl('instrument_knob_'+str(j),(-5.48+j*.25,1.73,.925),.014,.02,'bay_black',(math.pi/2,0,0))
    box('instrument_laptop_base',(-4.66,1.84,.91),(.28,.20,.013),'bay_steel')
    box('instrument_laptop_screen',(-4.66,1.94,1.02),(.28,.02,.22),'bay_black',(-.16,0,0))
    for j in range(2):ring('instrument_lead_coil_'+str(j)+'_',-5.10,1.89,.96+j*.01,.075,.003,'bay_wire')
    rod('instrument_power_lead',(-5.1,1.9,.95),(-4.6,1.84,.95),.003,'bay_wire')
    # Thin stock and tools visually connect the bench to the nearby red tool chest.
    for j in range(4):box('bench_tool_'+str(j),(-3.45,4.1+j*.22,.96),(.21,.07,.03),'bay_steel')
    box('bench_foam_strip',(-3.9,4.6,1.05),(.10,1.1,.025),'bay_white',(0,0,.12))
    # Walk-through refinement: actual wing test article is a deep lattice beam,
    # not a string of little ribs on two poles. Two freestanding trestles support it.
    bx,by=-4.25,.10
    for sy in (-.92,.92):
        for sx in (-1,1):
            rod(f'wing_trestle_leg_{sy}_{sx}',(bx+sx*.32,by+sy,.035),(bx+sx*.10,by+sy,1.05),.024,'bay_wood',True)
        box('wing_trestle_foot_'+str(sy),(bx,by+sy,.035),(.82,.08,.07),'bay_wood',collision=True)
        rod('wing_trestle_cross_'+str(sy),(bx-.25,by+sy,.49),(bx+.25,by+sy,.49),.018,'bay_wood')
    # Four corner longerons and repeated diagonal/trapezoid ribs, central plywood.
    for xx in (-.16,.16):
        for zz in (1.04,1.36):rod(f'wing_longeron_{xx}_{zz}',(bx+xx,by-1.56,zz),(bx+xx,by+1.56,zz),.009,'bay_pale')
    for j in range(13):
        yy=by-1.56+j*.26
        for xx in (-.16,.16):
            rod(f'wing_vertical_{j}_{xx}',(bx+xx,yy,1.04),(bx+xx,yy,1.36),.008,'bay_pale')
            if j<12:rod(f'wing_diagonal_{j}_{xx}',(bx+xx,yy,1.04),(bx+xx,yy+.26,1.36),.008,'bay_pale')
        rod('wing_top_rib_'+str(j),(bx-.16,yy,1.36),(bx+.16,yy,1.36),.008,'bay_pale')
        rod('wing_lower_rib_'+str(j),(bx-.16,yy,1.04),(bx+.16,yy,1.04),.008,'bay_pale')
    box('wing_plywood_center',(bx+.168,by+.10,1.19),(.012,.48,.30),'bay_pale')
    # Separate long pegboard/tool bench behind the fabrication island.
    tx,ty=-5.03,6.62
    box('toolbench_top',(tx,ty,.88),(.88,2.75,.05),'bay_white',collision=True)
    box('toolbench_lower',(tx,ty,.20),(.76,2.62,.04),'bay_steel',collision=True)
    for sx in (-1,1):
        for sy in (-1,1):box(f'toolbench_leg_{sx}_{sy}',(tx+sx*.37,ty+sy*1.28,.44),(.055,.055,.86),'bay_steel',collision=True)
    box('tool_pegboard',(tx-.43,ty,1.35),(.045,2.65,.85),'bay_wood',collision=True)
    # Rows of perforations and strongly colored hanging holders/clamps.
    for j in range(15):
        yy=ty-1.20+j*.17
        for k in range(4):cyl(f'peg_hole_{j}_{k}',(tx-.401,yy,1.09+k*.16),.008,.003,'bay_black',(0,math.pi/2,0))
        if j%2==0:
            box('tool_holder_'+str(j),(tx-.35,yy,1.39),(.13,.09,.19),'bay_lime')
            rod('tool_shaft_'+str(j),(tx-.25,yy,1.33),(tx-.25,yy,1.71),.010,'bay_silver')
            box('tool_handle_'+str(j),(tx-.25,yy,1.63),(.055,.05,.13),'bay_cyan' if j%4 else 'bay_red')
    # Parts drawer towers at each end, clear trays and piled loose tools.
    for sy in (-1,1):
        yy=ty+sy*1.01
        box('parts_drawer_tower_'+str(sy),(tx+.02,yy,1.12),(.34,.31,.45),'bay_black')
        for j in range(6):
            box(f'parts_drawer_{sy}_{j}',(tx+.20,yy,.96+j*.065),(.012,.27,.05),'bay_silver')
            box(f'parts_pull_{sy}_{j}',(tx+.215,yy,.96+j*.065),(.02,.07,.012),'bay_black')
    for j in range(12):
        yy=ty-.85+j*.15
        box('loose_tool_'+str(j),(tx+.14,yy,.94),(.28,.045,.025),'bay_cyan' if j%3==0 else 'bay_silver',(0,0,.3*(j%3)))
    for j in range(3):
        box('clear_parts_tray_'+str(j),(tx+.05,ty-.45+j*.37,.96),(.37,.28,.065),'bay_paper')
    # Two recognizable open-frame desktop 3D printers on the rear side table.
    px,py=-5.02,9.12
    box('printer_bench',(px,py,.87),(.85,1.77,.045),'bay_pale',collision=True)
    for sx in (-1,1):
        for sy in (-1,1):box(f'printer_benchleg_{sx}_{sy}',(px+sx*.34,py+sy*.76,.43),(.045,.045,.85),'bay_black',collision=True)
    for j in range(2):
        yy=py-.44+j*.85
        box('printer_base_'+str(j),(px,yy,.96),(.45,.52,.13),'bay_black')
        box('printer_bed_'+str(j),(px,yy,1.07),(.34,.36,.025),'bay_silver')
        for sx in (-1,1):box(f'printer_tower_{j}_{sx}',(px+sx*.21,yy+.17,1.29),(.035,.04,.55),'bay_black')
        rod('printer_toprail_'+str(j),(px-.21,yy+.17,1.56),(px+.21,yy+.17,1.56),.018,'bay_black')
        rod('printer_xrail_'+str(j),(px-.20,yy+.17,1.32),(px+.20,yy+.17,1.32),.012,'bay_silver')
        box('printer_head_'+str(j),(px,yy+.12,1.30),(.10,.09,.10),'bay_cyan')
        cyl('filament_spool_'+str(j),(px-.27,yy+.1,1.18),.11,.06,'bay_black',(0,math.pi/2,0))
    # Lattice-wing building work island with adhesives, bottles and paper.
    # The walkthrough reveals a second cluttered build surface north of the
    # earlier aircraft/display bench, not a bare single airframe table.
    ax,ay=-3.25,7.40
    box('build_island_top',(ax,ay,.91),(1.65,2.10,.06),'bay_black',collision=True)
    for sx in (-1,1):
        for sy in (-1,1):box(f'build_island_leg_{sx}_{sy}',(ax+sx*.72,ay+sy*.94,.45),(.065,.065,.88),'bay_steel',collision=True)
    box('build_island_lower',(ax,ay,.20),(1.48,1.87,.045),'bay_steel',collision=True)
    for sx in (-1,1):rod('build_wing_spar'+str(sx),(ax+sx*.21,ay-.95,1.02),(ax+sx*.21,ay+.95,1.02),.009,'bay_pale')
    for j in range(16):
        yy=ay-.95+j*.126
        rod('build_wing_rib'+str(j),(ax-.25,yy,1.01),(ax+.25,yy,1.04),.008,'bay_pale')
        if j<15:rod('build_wing_diag'+str(j),(ax-.21,yy,1.02),(ax+.21,yy+.126,1.02),.006,'bay_pale')
    for j in range(8):
        xx=ax-.62+(j%2)*1.20;yy=ay-.75+(j//2)*.46
        cyl('adhesive_bottle_'+str(j),(xx,yy,1.03),.035,.18,'bay_white' if j%2 else 'bay_red')
        cyl('bottle_cap_'+str(j),(xx,yy,1.13),.025,.025,'bay_lime')
        box('paper_scrap_'+str(j),(xx+.11,yy+.08,.947),(.19,.15,.004),'bay_paper',(0,0,j*.28))
    cyl('tall_build_bottle',(ax+.57,ay+.28,1.17),.055,.42,'bay_white')
    box('brown_supply_bag',(ax+.55,ay+.76,1.15),(.24,.25,.40),'bay_wood')
    # Correct the small supported RC plane to the distinctive red-white-blue livery.
    box('rc_blue_fuselage',(-3.35,3.13,.865),(.71,.06,.035),'bay_cyan',(0,.035,-.12))
    for sy in (-1,1):
        box('rc_blue_wingtip'+str(sy),(-3.40,3.13+sy*.40,.882),(.25,.10,.009),'bay_cyan',(0,0,-.12))
    # A larger red toolchest with an open top lid and pulled-out lower drawer.
    cx,cy=-2.1,4.7
    box('open_toolchest_body',(cx,cy,.58),(.76,.51,1.04),'bay_red',collision=True)
    for j in range(9):
        zz=.18+j*.085
        box('open_toolchest_drawer_'+str(j),(cx,cy-.265,zz),(.70,.025,.065),'bay_red')
        box('open_toolchest_pull_'+str(j),(cx,cy-.285,zz+.015),(.58,.025,.015),'bay_silver')
    box('toolchest_top_tray',(cx,cy,1.12),(.74,.49,.08),'bay_black')
    box('toolchest_open_lid',(cx,cy+.19,1.38),(.76,.035,.40),'bay_red',(-.18,0,0))
    for j in range(3):box('toolchest_lid_badge_'+str(j),(cx-.22+j*.22,cy+.16,1.38),(.16,.008,.07),'bay_silver')
    box('toolchest_open_bottom',(cx,cy-.39,.10),(.69,.37,.07),'bay_black',collision=True)
    box('toolchest_bottom_front',(cx,cy-.59,.14),(.71,.025,.16),'bay_red')
    box('toolchest_gloves',(cx-.15,cy-.40,.17),(.22,.17,.07),'bay_white')
    # Walking aisle views the drawer fronts from the east.
    for part in parts:
        if 'toolchest_' in part['name']:
            x,y,z=part['pos'];dx=x-cx;dy=y-cy
            part['pos']=(cx-dy,cy+dx,z)
            rx,ry,rz=part['rot'];part['rot']=(rx,ry,rz+math.pi/2)
    return parts
