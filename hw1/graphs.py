import pandas as pd
import matplotlib.pyplot as plt


df_cz = pd.read_csv("tables/CZ_char_mess.txt", sep="\t")
ax = df_cz.plot.bar(x='likelihood', y=['min_entropy', 'max_entropy', 'avg_entropy'], rot=0, legend=True)
ax.set_ylabel("Conditional entropy")
ax.set_xlabel("Mess up Likelihood")
ax.set_title("Czech characters")
ax.set_ylim(4, 5)
plt.savefig("graphs/CZ_char_mess.png")
# plt.show()


df_en = pd.read_csv("tables/EN_char_mess.txt", sep="\t")
ax = df_en.plot.bar(x='likelihood', y=['min_entropy', 'max_entropy', 'avg_entropy'], rot=0, legend=True)
ax.set_ylabel("Conditional entropy")
ax.set_xlabel("Mess up Likelihood")
ax.set_title("English characters")
ax.set_ylim(4.7, 5.4)
plt.savefig("graphs/EN_char_mess.png")
# plt.show()




df_cz = pd.read_csv("tables/CZ_word_mess.txt", sep="\t")
ax = df_cz.plot.bar(x='likelihood', y=['min_entropy', 'max_entropy', 'avg_entropy'], rot=0, legend=True)
ax.set_ylabel("Conditional entropy")
ax.set_xlabel("Mess up Likelihood")
ax.set_title("Czech words")
ax.set_ylim(4.8, 5)
plt.savefig("graphs/CZ_word_mess.png")
# plt.show()


df_en = pd.read_csv("tables/EN_word_mess.txt", sep="\t")
ax = df_en.plot.bar(x='likelihood', y=['min_entropy', 'max_entropy', 'avg_entropy'], rot=0, legend=True)
ax.set_ylabel("Conditional entropy")
ax.set_xlabel("Mess up Likelihood")
ax.set_title("English words")
ax.set_ylim(5.3, 5.6)
plt.savefig("graphs/EN_word_mess.png")
# plt.show()



df_en = pd.read_csv("tables/EN_adj_lambda_stats.txt", sep="\t")
df_en['adjust'] = df_en['adjust'].fillna('Initial')
ax = df_en.plot.bar(x='percent', y='cross_entropy', color=df_en['adjust'].map({'Initial': 'gray', 'decrease': 'red', 'increase': 'green'}), legend=False)
ax.set_xlabel("Percent")
ax.set_ylabel("Cross Entropy")
ax.set_title("English lambda adjustments")
ax.set_ylim(4, 8)
plt.savefig("graphs/EN_adj_lambda_stats.png")
# plt.show()


df_cz = pd.read_csv("tables/CZ_adj_lambda_stats.txt", sep="\t")
df_cz['adjust'] = df_cz['adjust'].fillna('Initial')
ax = df_cz.plot.bar(x='percent', y='cross_entropy', color=df_cz['adjust'].map({'Initial': 'gray', 'decrease': 'red', 'increase': 'green'}), legend=False)
ax.set_xlabel("Percent")
ax.set_ylabel("Cross Entropy")
ax.set_title("Czech lambda adjustments")
ax.set_ylim(8, 12)
plt.savefig("graphs/CZ_adj_lambda_stats.png")
# plt.show()