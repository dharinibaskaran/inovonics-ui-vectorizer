
from pycocotools.coco import COCO
import json
from inovonics_ann_builder import InovonicsAnnotationBuilder as InovAnnBuild
from from_root import from_root

def main(coco_source_path, inovonics_anns_dest_path, img_ids=[1]):
    coco = COCO(coco_source_path)
    if(len(img_ids) == 1):
        for img_id in img_ids:
            coco_img_to_inovonics_json(coco, inovonics_anns_dest_path, img_id)
    else:
        for img_id in img_ids:
            coco_img_to_inovonics_json(coco, inovonics_anns_dest_path[0:-5]+"_"+str(img_id)+".json", img_id)

    
def coco_img_to_inovonics_json(coco, inovonics_anns_dest_path, img_id=0):
    #iterate
    annotation_full_file = []
    count = 0
    for ann in coco.imgToAnns[img_id]:
        print(ann)
        inov_ann_build = InovAnnBuild()
        inov_ann_build.set_id(str(count))
        inov_ann_build.set_body("Room "+str(count))
        inov_ann_build.set_type("Selection")
        inov_ann_build.set_target("FragmentSelector","http://www.w3.org/TR/media-frags/",ann["bbox"])
        annotation_full_file.append(inov_ann_build.final_output())
        count+=1

    #save file
    coco_file = open(inovonics_anns_dest_path,'w')
    json.dump(annotation_full_file,coco_file,indent=4)
    coco_file.close()