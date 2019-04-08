
# 7 blocks
for b in 1 8 9 15 76 89 105; do
  python -u preprocess_data.py $HOME/data EC2 "$b"
done

# 7 blocks
for b in 15 39 46 49 53 60 63; do
  python -u preprocess_data.py $HOME/data EC9 "$b"
done

# 14 blocks
for b in 1 2 4 6 9 21 63 65 67 69 71 78 82 83; do
  python -u preprocess_data.py $HOME/data GP31 "$b"
done

# 3 blocks
for b in 1 5 30; do
  python -u preprocess_data.py $HOME/data GP33 "$b"
done
