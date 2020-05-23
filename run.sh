echo '请确定数据集都放在了~/data/目录下，下面创建软链接...'
cd data
ln -s ~/data/cifar-10-batches-py cifar-10-batches-py
cd ..

echo '综和比较'
for file in experiments/*
do
    python  train.py --config $file
done