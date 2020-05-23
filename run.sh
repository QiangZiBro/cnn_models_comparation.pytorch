echo '综和比较...'
for file in experiments/*
do
    python  train.py --config $file
done
