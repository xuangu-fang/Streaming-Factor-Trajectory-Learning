for i in 3 5 7;
do
    # Beijing 15k
    # python dynamic_streaming_CP.py --dataset=beijing_15k --R_U=$i --num_fold=5  --machine=$USER 
    # python dynamic_streaming_tucker.py --dataset=beijing_15k --R_U=$i --num_fold=5  --machine=$USER 

    # # Beijing 20k
    # python dynamic_streaming_CP.py --dataset=beijing_20k --R_U=$i --num_fold=5 --machine=$USER
    # python dynamic_streaming_tucker.py --dataset=beijing_20k --R_U=$i --num_fold=5 --machine=$USER

    # server
    # python dynamic_streaming_CP.py --dataset=server --R_U=$i --num_fold=5  --machine=$USER
    # python dynamic_streaming_tucker.py --dataset=server --R_U=$i --num_fold=5  --machine=$USER

    # # # fitRecord_50k
    # python dynamic_streaming_CP.py --dataset=fitRecord --R_U=$i --num_fold=5  --machine=$USER
    # python dynamic_streaming_tucker.py --dataset=fitRecord --R_U=$i --num_fold=5  --machine=$USER
done

