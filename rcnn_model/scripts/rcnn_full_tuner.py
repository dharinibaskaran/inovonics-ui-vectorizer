from rcnn_config import write_config
import rcnn_train
import rcnn_eval
import rcnn_run

run_iter_counts = [3000,4000,5000,6000,7000,8000,9000,10000]

def main():
    cfg = write_config()
    for i in range(0,len(run_iter_counts)):
        iters = run_iter_counts[i]
        if(iters > 0):
            cfg.SOLVER.MAX_ITER = iters
            cfg.SOLVER.STEPS = (int(iters*.5),int(iters*.75))
            rcnn_train.main(cfg)
            rcnn_run.default_sample(cfg, i)
            rcnn_eval.main(cfg,"eval_results_run_"+str(i)+".txt")

main()