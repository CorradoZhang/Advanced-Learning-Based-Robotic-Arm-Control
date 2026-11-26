"""Camera-only AprilTag preview; does not import robot or motor drivers."""
import argparse
import cv2
from vision.apriltag.detector import TagObservations


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--intrinsics',required=True)
    p.add_argument('--camera-index',type=int,default=0)
    p.add_argument('--backend',type=int,default=0,help='OpenCV backend; 1200 for AVFoundation')
    p.add_argument('--tag-size-m',type=float,default=.0215)
    p.add_argument('--tag-id',type=int,default=0)
    args=p.parse_args()
    detector=TagObservations(args.intrinsics,args.tag_size_m,args.tag_id)
    cap=cv2.VideoCapture(args.camera_index,args.backend)
    width,height=detector.calibration['image_size']
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
    try:
        if not cap.isOpened(): raise RuntimeError('Cannot open selected camera')
        while True:
            ok,bgr=cap.read()
            if not ok: raise RuntimeError('Camera stopped delivering frames')
            rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            obs=detector.extract(rgb)
            valid=bool(obs['observation.apriltag.valid'][0])
            display=detector.annotate(rgb,valid,obs)
            xyz=obs['observation.apriltag.position']
            text=f'XYZ m: {xyz[0]:+.3f} {xyz[1]:+.3f} {xyz[2]:+.3f}' if valid else 'Tag unavailable: valid=0'
            cv2.putText(display,text,(10,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,0),2)
            cv2.imshow('AprilTag camera preview - Esc to exit',cv2.cvtColor(display,cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1)&255 in (27,ord('q')): break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__=='__main__': main()
