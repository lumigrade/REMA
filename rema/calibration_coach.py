"""Software-rendered mirrored camera cues and an animated 3D movement coach.
No sensor geometry is altered; every flip is applied to display copies only.
"""
import math
import cv2
import numpy as np
from .calibration_guide import mirror_points

ORANGE=(60,165,255)
GREEN=(140,224,94)
BLUE=(255,185,85)
WHITE=(237,239,243)
EDGE_IDS={'LEFT':(1,2),'RIGHT':(0,3),'TOP':(0,1),'BOTTOM':(3,2)}


def text(im, value, xy, size=.62, color=WHITE, thick=1):
    cv2.putText(im,value,xy,cv2.FONT_HERSHEY_SIMPLEX,size,(18,20,24),thick+3,cv2.LINE_AA)
    cv2.putText(im,value,xy,cv2.FONT_HERSHEY_SIMPLEX,size,color,thick,cv2.LINE_AA)


def selected_edge(outline, edge):
    # Use observed screen-side edges, so modest in-plane rotation is harmless.
    points=mirror_points(outline)
    pairs=[(points[i],points[(i+1)%4]) for i in range(4)]
    centers=np.array([(a+b)/2 for a,b in pairs])
    axis=0 if edge in ('LEFT','RIGHT') else 1
    index=int(np.argmin(centers[:,axis]) if edge in ('LEFT','TOP') else np.argmax(centers[:,axis]))
    return pairs[index]


def mirror_preview(raw,outline,center,pose,cue,progress=0):
    result=cv2.flip(raw,1)  # labels are drawn AFTER flipping so text stays readable
    if outline is not None:
        pts=mirror_points(outline).astype(np.int32)
        cv2.polylines(result,[pts],True,GREEN if cue.get('kind')=='hold' else WHITE,2,cv2.LINE_AA)
        if cue.get('kind')=='tilt':
            a,b=selected_edge(outline,cue['edge']);a=tuple(np.rint(a).astype(int));b=tuple(np.rint(b).astype(int))
            cv2.line(result,a,b,(25,32,40),13,cv2.LINE_AA)
            cv2.line(result,a,b,ORANGE,8,cv2.LINE_AA)
            mid=(np.array(a)+b)/2
            cv2.circle(result,a,8,ORANGE,-1,cv2.LINE_AA);cv2.circle(result,b,8,ORANGE,-1,cv2.LINE_AA)
            x=int(np.clip(mid[0]-65,12,450));y=int(np.clip(mid[1]-14,75,430))
            text(result,cue['edge']+' EDGE',(x,y),.64,ORANGE,2)
    if pose and cue.get('kind')=='position':
        target=tuple(np.rint(mirror_points(pose.center)).astype(int))
        cv2.circle(result,target,23,BLUE,3,cv2.LINE_AA)
        if center is not None:
            current=tuple(np.rint(mirror_points(center)).astype(int))
            cv2.arrowedLine(result,current,target,BLUE,4,cv2.LINE_AA,tipLength=.25)
    if cue.get('kind')=='hold' and center is not None:
        c=tuple(np.rint(mirror_points(center)).astype(int))
        cv2.ellipse(result,c,(30,30),-90,0,max(1,progress*360),GREEN,5,cv2.LINE_AA)
    text(result,'MIRROR VIEW - move like your reflection',(14,28),.55,BLUE)
    return result


def board_points(tilt):
    tx,ty=np.radians(tilt)
    rotation=cv2.Rodrigues(np.array([-ty,tx,0.]))[0]
    corners=np.array([[-.1485,-.105,0],[.1485,-.105,0],[.1485,.105,0],[-.1485,.105,0]])
    # Scene coordinates are user's right / down / away toward the camera.
    return (corners@rotation.T)*np.array([-1,1,-1])


def motion_points(tilt,start,edge):
    points=board_points(tilt)
    opposite={'LEFT':'RIGHT','RIGHT':'LEFT','TOP':'BOTTOM','BOTTOM':'TOP'}[edge]
    ids=list(EDGE_IDS[opposite])
    return points+(board_points(start)[ids].mean(axis=0)-points[ids].mean(axis=0))


def project(points):
    view=cv2.Rodrigues(np.array([-.43,.58,0.]))[0]
    p=points@view.T
    scale=690/(1+p[:,2])
    return np.c_[300+p[:,0]*scale,215+p[:,1]*scale].astype(np.int32)


def draw_board(im,tilt,edge=None,anchor_tilt=None):
    points=board_points(tilt) if anchor_tilt is None else motion_points(tilt,anchor_tilt,edge);front=project(points);back=project(points+np.array([0,0,.009]))
    # A shaded solid with thickness and a pattern supplies perspective cues.
    cv2.fillConvexPoly(im,back,(58,70,84),cv2.LINE_AA)
    for a,b in [(0,1),(1,2),(2,3),(3,0)]:
        cv2.fillConvexPoly(im,np.array([front[a],front[b],back[b],back[a]]),(88,105,124),cv2.LINE_AA)
    cv2.fillConvexPoly(im,front,(219,228,234),cv2.LINE_AA)
    for y in range(4):
        for x in range(6):
            if (x+y)%2:continue
            def at(u,v):return points[0]*(1-u)*(1-v)+points[1]*u*(1-v)+points[2]*u*v+points[3]*(1-u)*v
            cell=np.array([at(x/6,y/4),at((x+1)/6,y/4),at((x+1)/6,(y+1)/4),at(x/6,(y+1)/4)])
            cv2.fillConvexPoly(im,project(cell),(103,126,145),cv2.LINE_AA)
    cv2.polylines(im,[front],True,WHITE,2,cv2.LINE_AA)
    if edge:
        a,b=EDGE_IDS[edge];cv2.line(im,tuple(front[a]),tuple(front[b]),ORANGE,9,cv2.LINE_AA)
        for i in (a,b):cv2.circle(im,tuple(front[i]),7,ORANGE,-1,cv2.LINE_AA)
    return points


def coach_panel(pose,cue,obs,seconds,progress=0,depth=None):
    im=np.full((480,640,3),(35,30,26),np.uint8)
    kind=cue.get('kind','wait')
    stages=['DISTANCE','POSITION','TILT','HOLD']
    for i,name in enumerate(stages):
        x=12+i*157;active=kind==name.lower()
        cv2.rectangle(im,(x,9),(x+148,43),GREEN if active else (69,63,55),-1)
        text(im,f'{i+1} {name}',(x+8,32),.52,WHITE,1)
    if pose is None:
        text(im,'FOLLOW ONE INSTRUCTION AT A TIME',(34,103),.70,BLUE,2)
        draw_board(im,[0,0])
        text(im,'1  Set the distance',(28,357),.7)
        text(im,'2  Slide the whole board',(28,391),.7)
        text(im,'3  Push / pull ONE orange edge',(28,425),.7,ORANGE)
        text(im,'4  Hold still - capture is automatic',(28,459),.65,GREEN)
        return im
    if kind=='tilt':
        edge=cue['edge'];away=cue['away'];current=np.array(cue['current'],float);goal=current.copy()
        axis=cue['axis'];goal[axis]=cue['target'][axis]
        # Pause at each end of the demonstration, then replay. Current pose is
        # the real measured tilt; the animation corrects only the requested axis.
        cycle=(seconds%3.6)/3.6
        amount=float(np.clip((cycle-.12)/.64,0,1));amount=amount*amount*(3-2*amount)
        tilt=current+(goal-current)*amount
        text(im,edge+' EDGE: '+('PUSH AWAY' if away else 'PULL TOWARD YOU'),(24,78),.82,ORANGE,2)
        text(im,'Keep the opposite edge steady',(24,109),.62)
        pts=draw_board(im,tilt,edge,current)
        text(im,'3D MOTION EXAMPLE - repeats',(28,322),.56,BLUE)
        # A separate orthographic view disambiguates depth, with camera and user
        # explicitly marked. Native normal.x changes sign under the mirror.
        left,right=((0,3),(1,2)) if axis==0 else ((0,1),(3,2))
        def profile(angles):
            p=motion_points(angles,current,edge)
            ends=np.array([p[list(left)].mean(axis=0),p[list(right)].mean(axis=0)])
            transverse=ends[:,0] if axis==0 else ends[:,1]
            return np.c_[292+transverse*550,410-ends[:,2]*260].astype(np.int32)
        actual=profile(current);target=profile(goal);animated=profile(tilt)
        cv2.line(im,tuple(actual[0]),tuple(actual[1]),(115,121,127),5,cv2.LINE_AA)
        cv2.line(im,tuple(target[0]),tuple(target[1]),GREEN,3,cv2.LINE_AA)
        cv2.line(im,tuple(animated[0]),tuple(animated[1]),WHITE,5,cv2.LINE_AA)
        edge_index=(1 if edge=='LEFT' else 0) if axis==0 else (0 if edge=='TOP' else 1)
        point=animated[edge_index];cv2.circle(im,tuple(point),8,ORANGE,-1,cv2.LINE_AA)
        arrow_end=point+np.array([0,-35 if away else 35])
        cv2.arrowedLine(im,tuple(point),tuple(arrow_end),ORANGE,4,cv2.LINE_AA,tipLength=.35)
        text(im,'CAMERA',(245,357),.56,BLUE,1)
        text(im,'YOU',(275,470),.56,BLUE,1)
        text(im,'TOP VIEW' if axis==0 else 'SIDE VIEW',(16,373),.46)
        text(im,'green = goal',(16,453),.42,GREEN)
        for i,end in enumerate(animated):
            name=(('RIGHT','LEFT')[i] if axis==0 else ('TOP','BOTTOM')[i])
            text(im,name,tuple(end+[-28,24]),.36)
    elif kind=='position':
        direction=cue['direction']
        text(im,'SLIDE '+direction,(32,86),1.0,BLUE,2)
        text(im,'Move both hands together. Keep the angle.',(24,125),.62)
        draw_board(im,obs.get('tilt',[0,0]))
        direction_vector={'LEFT':(-95,0),'RIGHT':(95,0),'UP':(0,-65),'DOWN':(0,65)}[direction]
        start=np.array([300,225]);end=start+direction_vector
        cv2.arrowedLine(im,tuple(start),tuple(end),BLUE,9,cv2.LINE_AA,tipLength=.35)
        text(im,'On the mirror: move the board centre',(26,367),.68)
        text(im,'into the BLUE circle.',(26,403),.75,BLUE,2)
        text(im,'No rotating or twisting in this step.',(26,447),.6)
    elif kind=='distance':
        away=cue['away']
        text(im,'WHOLE BOARD: '+('PUSH AWAY' if away else 'PULL TOWARD YOU'),(20,83),.72,ORANGE,2)
        text(im,f'NOW {cue["current"]:.2f} m     GOAL {cue["target"]:.2f} m',(42,125),.83,BLUE,2)
        draw_board(im,obs.get('tilt',[0,0]))
        text(im,'CAMERA',(42,373),.63,BLUE)
        text(im,'YOU',(480,373),.63,BLUE)
        start=(440,400) if away else (150,400);end=(150,400) if away else (440,400)
        cv2.arrowedLine(im,start,end,ORANGE,9,cv2.LINE_AA,tipLength=.18)
        text(im,'Move both hands together; keep it facing camera.',(20,455),.54)
    elif kind=='hold':
        text(im,'THAT IS RIGHT - HOLD STILL',(30,89),.85,GREEN,2)
        draw_board(im,obs.get('tilt',[0,0]))
        cv2.rectangle(im,(52,365),(588,396),(65,64,56),-1)
        cv2.rectangle(im,(52,365),(52+round(536*progress),396),GREEN,-1)
        text(im,'Automatic capture in '+f'{max(0,1.4*(1-progress)):.1f}'+' s',(65,439),.8,GREEN,2)
    else:
        text(im,'SHOW THE WHOLE PRINT',(48,95),.86,BLUE,2)
        draw_board(im,[0,0])
        text(im,'Keep it landscape, facing the camera.',(28,373),.66)
        text(im,'Hold clear of your chest / background.',(28,414),.63)
        text(im,'No need to match a floating wire outline.',(28,455),.60)
    if depth is not None:
        inset=cv2.resize(cv2.flip(depth,1),(100,75))
        im[200:275,525:625]=inset
        text(im,'DEPTH',(535,193),.34)
    return im
