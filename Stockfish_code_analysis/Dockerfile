FROM ubuntu:20.04  
LABEL maintainer="theomatricon@gmail.com" 

SHELL ["/bin/bash", "--login", "-c"]
# Update and install wget
RUN apt-get -y update && apt-get -y install wget build-essential
# Install conda
RUN mkdir -p ~/miniconda3 && \
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh && \
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 && \
rm -rf ~/miniconda3/miniconda.sh  && \
~/miniconda3/bin/conda init bash && \
~/miniconda3/bin/conda create -n stockfish python==3.9
SHELL ["root/miniconda3/bin/conda", "run" ,"-n", "stockfish", "/bin/bash", "--login", "-c"]
# Install python dependencies
RUN pip install six && \
pip install numpy==1.21.1 tqdm==4.64.0 pandas
# save Stockfish
COPY . .
# Make the first version fo Stockfish
RUN cp ./src/movegen_other.cpp ./src/movegen.cpp && \
cd ./src  && \
make -j profile-build ARCH=x86-64 && \
mv ./stockfish ../stockfish_sorted
# Make the second version fo Stockfish
RUN cp ./src/movegen_other_sym.cpp ./src/movegen.cpp && \
cd ./src  && \
make -j profile-build ARCH=x86-64 && \
mv ./stockfish ../stockfish_mirror
# Make the third version fo Stockfish
RUN cp ./src/types_axis.h ./src/types.h && \
cd ./src  && \
make -j profile-build ARCH=x86-64 && \
mv ./stockfish ../stockfish_axis
ENTRYPOINT ["root/miniconda3/bin/conda", "run" ,"-n", "stockfish", "--no-capture-output", "./run_all_experiments.sh"]

