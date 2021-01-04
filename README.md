# Self Organizing Maps (SOM): segmenting data

### Self organizing maps segmenting Iris data, oil patch descriptions, and national wealth by features.      

An useful application of SOM is to organize objects by similarity of features.  Once objects are organized by similarity of features, we are able to make interpretations based on dimensions, distance, and display.  While most applications focus on segmentation, we are also able to interpret results based on dimensions as well as distance/location.  We also illustrate how Python libraries may be used to present changes in segment membership as we approach a solution.     

We will describe:
<ol type="1">
	<li>different kinds of data used for illustration</li>
		- classic Iris data (3x3x4)
		- oil patch data (3x3x4)
		- wealth of nations (7x7)
	<li>measures of distance and interpretation</li>
	<li>dimensions and interpretations of dimensions</li>
	<li>observing the changes in assignment of objects to segments</li>
	<li>a Python program which provides a basis for modifying data sources, measures of distance, and number of dimensions</li>
	<li>an interpretation of results</li>
</ol>

There are three examples of outcomes from the use of SOM below.  It's worth examining these results and discussing them to see the utility of these kinds of segmentation.  

The first illustrates the results for the classis Iris data.  The data is classic for two reasons.  First, it is used to demonstrate how good an algorithm is at grouping or segmenting objects in comparison to other algorithms.  Second, the Iris genus has about 300 species, of which versicolor is an allopolyploidy of setosa and virginica - multiple genomes from different species operating within one plant.  The result is that any segmentation may reflect ploids found within Iris versicolor.  

For this matrix, I provide background colors and textual displays within each cells as standard output for this Python stream - the stars and summaries on the left are additional decorations.  The matrix consists of 3 dimensions measuring 3 by 3 by 4 cells.  At the bottom of the matrix is the order of the four most important measures:  petal width, petal length, sepal length, and sepal width.  The RGB colors reflect combination of the three most important measures.  The colors are grayish reflecting the fact that the top three measures are highly correlated with one another - after all, physical sizes are generally correlated with one another.  

In doing segmentation, I generally prefer SOM over G-means.  In my experience, G-means has a tendency to produce few groups which have a very uneven distribution of objects.  As an example, there might be 2% in one group 5% in a second, and 93% in a third group.  While 3 groups might be reasonable for the classic Iris data, the display of SOM provides an interpretable basis for deciding if any particular number of groups is really the best.  Beyond this, using multi-dimensional SOM allows for interpretation of dimensions.  

<p align="center">
<img src="/images/Simplex3x3x4_iris_Data.jpg" width="400" height="450">
</p>

The second segmentation illustrates results for oil patch data from around the world.  In contrast to the previous table, the colors are more intense reflecting the fact that the dimensions are more independent of one another.  On the left we have a summary of oil patches and their position in the matrix. At the bottom are the four most important measures:  downhole pump, pipeline, water injection, and development intensity.  

<p align="center">
<img src="/images/Simplex3x3x4_oilPatch.jpg" width="400" height="450">
</p>

Next, we observe the results for the classic SOM analysis of national wealth.  For this we use a 7 by 7 matrix linked with a map of the world. What is unique about this solution is we are able to observe the iterations to solution on both the matrix and the map at the same time using libraries available in Python.  It's important to note that there are multiple solutions when grouping objects and it's possible that investigators may obtain insights by watching the convergence to a solution.  

<p align="center">
<img src="/images/SelfOrganizing_42_120.jpg" width="400" height="200">
</p>

The convergence displayed below generally fits with other classifications of nations.    

<p align="center">
<img src="/images/SelfOrganizing_120_120.jpg" width="500" height="300">
</p>


### Motivation for creating Self Organizing Maps (SOMS):
Considerable effort has been applied to building neural nets to discriminate between patients who are healthy and those patients with pneumonia based on x-rays.  An avenue which 

### Citations:
<dl>
<dt>Jeff Heaton, Jeff Heaton's Deep Learning Course,</dt><dd> Available from https://www.heatonresearch.com/course/, accessed February, 2020. </dd>
<dt>Tory Walker, Histogram equalizer, </dt> <dd>Available from https://github.com/torywalker/histogram-equalizer, accessed March, 2020</dd>
<dt>maps production from, </dt><dd>https://mohammadimranhasan.com/geospatial-data-mapping-with-python/</dd>
<dt>maps using Python, </dt><dd>https://github.com/hasanbdimran/Mapping-in-python</dd>
<dt>using geoPandas, </dt><dd>https://medium.com/@james.moody/creating-html-choropleths-using-geopandas-2b8ced9f1632</dd>
<dt>chloropleth generators, </dt><dd>https://github.com/jmsmdy/html-choropleth-generator</dd>
</dl>

### Deliverables:
  1.  description of issues identified and resolved within specified limitations
  2.  code fragments illustrating the core of how an issue was resolved
  3.  three Python programs which illustrate the use of SOMs in multidimensional arrrays

### Limitations and caveates:

  1.  stream:  refers to the overall process of streaming/moving data through input, algorithms, and output of data and its evaluation.
  2.  convergence:  since there are no unique solutions in GAN, convergence is sufficient when there are no apparent improvements in a subjective evaluation of clarity of images being generated.   
  3.  limited applicability:  the methods described work for a limited set of data and cGan problems.
  4.  bounds of model loss:  there is an apparent relationship between mode collapse and model loss - when model loss is extreme (too high or too low) then there is mode collapse.  
  
### Software and hardware requirements:
    - Python version 3.7.3
        - Numpy version 1.17.3
        - Matplotlib version 3.0.3
	- geopandas version
	- os, sys, datetime, math, time
	- urllib, webbrowser
	- cartopy version 
    - Operating system used for development and testing:  Windows 10

#### The process:

 Creating three SOMs as illustration, I provide limited working solutions to the following problems:

<ol type="1">
  <li>classifying the class allopolyploidy Iris dataset in a 3D array</li>
  <li>classifying oilpatch data in a 3D array</li>
  <li>classifying nation's wealth and mapping iterations</li>
</ol>


The recommended folder structure looks as follows:
<ul>
    <li>embedding_derived_heatmaps-master (or any folder name)</li>
	<ul>
	<li> files (also contains three Python programs - program run from here)</li>
	<ul>
		<li> <b>SOM map</b></li>
		<ul>
			<li> <b>label_results</b> (contains five .h5 generator model files)</li>
		</ul>
		<li> <b>xray</b></li>
		<ul>
			<li> <b>label_results</b> (contains five .h5 generator model files)</li>
		</ul>
		<li> <b>cgan</b> (contains images from summary analysis of models)</li>
	</ul>
	<li> images (contains images for README file)</li>
	</ul>
</ul>

#### LICENSE  <a href="/LICENSE">MIT license</a>
