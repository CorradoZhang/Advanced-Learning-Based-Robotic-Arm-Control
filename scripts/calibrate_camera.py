"""Camera-only pinhole calibration: Space capture, Enter solve, Esc exit."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np


def solve(points,image_size,board,square_m):
    grid=np.zeros((board[0]*board[1],3),np.float32)
    grid[:,:2]=np.mgrid[0:board[0],0:board[1]].T.reshape(-1,2)*square_m
    rms,matrix,distortion,_,_=cv2.calibrateCamera([grid.copy() for _ in points],points,image_size,None,None)
    if not np.isfinite(matrix).all() or not np.isfinite(distortion).all():
        raise ValueError('Calibration failed; capture more varied board poses')
    return dict(camera_matrix=matrix.tolist(),distortion_coefficients=distortion.ravel().tolist(),
        image_size=list(image_size),rms_reprojection_error_px=float(rms),
        board_inner_corners=list(board),square_size_m=square_m,views=len(points),model='pinhole')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--camera-index',type=int,default=0)
    p.add_argument('--backend',type=int,default=0)
    p.add_argument('--width',type=int,default=640)
    p.add_argument('--height',type=int,default=480)
    p.add_argument('--columns',type=int,default=9,help='Number of INNER corners')
    p.add_argument('--rows',type=int,default=6,help='Number of INNER corners')
    p.add_argument('--square-mm',type=float,default=20.)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if min(args.width,args.height,args.columns,args.rows)<2 or not np.isfinite(args.square_mm) or args.square_mm<=0:
        p.error('Image/board dimensions and square size must be positive')
    if args.out.exists(): p.error('Output exists; choose a new calibration file')
    cap=cv2.VideoCapture(args.camera_index,args.backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,args.height)
    points=[]
    board=(args.columns,args.rows)
    try:
        if not cap.isOpened(): raise RuntimeError('Cannot open selected camera')
        while True:
            ok,bgr=cap.read()
            if not ok: raise RuntimeError('Camera stopped delivering frames')
            if bgr.shape[:2] != (args.height,args.width):
                raise ValueError('Camera resolution differs from requested calibration resolution')
            gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
            found,corners=cv2.findChessboardCornersSB(gray,board)
            display=bgr.copy()
            if found: cv2.drawChessboardCorners(display,board,corners,found)
            cv2.putText(display,f'Views: {len(points)} | Space capture | Enter save | Esc exit',
                (10,25),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0),1)
            cv2.imshow('Camera intrinsics calibration',display)
            key=cv2.waitKey(1)&255
            if key==27: break
            if key==32 and found: points.append(corners.copy())
            if key in (10,13):
                if len(points)<10:
                    print('Capture at least 10 diverse views; 20 or more recommended.',flush=True)
                    continue
                result=solve(points,(args.width,args.height),board,args.square_mm/1000.)
                result['camera_index']=args.camera_index
                args.out.parent.mkdir(parents=True,exist_ok=True)
                args.out.write_text(json.dumps(result,indent=2))
                print(f'Saved {args.out}; RMS={result["rms_reprojection_error_px"]:.3f} px')
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__=='__main__': main()
