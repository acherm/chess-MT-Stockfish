# FILES="reals/ev_lichess2000-2500-10_nocastlingsim_mirror_d_20.pkl reals/ev_lichessAllElos-10_nocastlingsim_mirror_d_20.pkl reals/ev_Carlsen-10sim_mirror_d_20.pkl"
# FILES="reals/ev_Carlsen-10sim_mirror_d_20.pkl sim_mirror_d=10/evaluations50000_sim_mirror_d_10_v16.pkl reals/ev_lichess2000-2500-20_nocastlingsim_mirror_d_20.pkl"
FILES="$2"
IS_MIRROR="$3"
PREFIX=""
DEPTHS="10 20"
PROCS=$1
mkdir -p output
echo "IS_MIRROR: $IS_MIRROR"
for file in $FILES;
do
    for depth in $DEPTHS;
    do
        name=$(basename $file)
        python src/test_stockfish.py $depth $PROCS "$PREFIX/$file" "$IS_MIRROR"
        mv ./stockfish_$depth.json output/${name}_${depth}.json
    done
done
