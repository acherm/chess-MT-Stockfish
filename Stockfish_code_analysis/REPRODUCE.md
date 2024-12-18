
Clone the stockfish repository.

```bash
git clone https://github.com/official-stockfish/Stockfish.git
```

Move everything from this artifact folder at the root of the Stockfish repo.
```bash
mv artifact/* Stockfish/
```

Now build the docker image.

```bash
echo "RUN IMAGE ===================================="
echo "ARGS are: <depth> <file.pkl> <is_mirror>"
docker run --name sf_container stockfish:1.0 16 sim_axis_d=10/evaluations50000_sim_axis_d_10.pkl 1
echo "EXTRACT FILE ================================="
docker cp sf_container:./output .
```

You should have a bunch of JSON files which you can exploit with:

```bash
python exploit_stockfish.py myfile.json
```

(The script was originaly in the folder "artifact").