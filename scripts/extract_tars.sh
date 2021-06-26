for dir in ./*/
do
    cd $dir
    for a in *.tar.gz
    do
        tar -xvzf $a --one-top-level
    done
    cd ..
done