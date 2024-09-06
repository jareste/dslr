import matplotlib.pyplot as plt

#test for ploting

plt.plot([1, 2, 3, 4], [10, 20, 25, 30], label="Line")
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.title('Simple Line Plot')
plt.legend()

plt.savefig('/output/plot.png')

# im into a docker of course it will not be shown, but i can see it on output/plot.png
plt.show()
