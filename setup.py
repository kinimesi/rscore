from distutils.core import setup

setup(
	name='rscore',
	version='0.90',
	packages=['rscore'],
	url='https://github.com/kinimesi/rscore',
	license='Apache License v2.0',
	author='tinko minko',
	author_email='',
	description='text readability level classifier',
	install_requires=[
          'nltk == 3.0.0',
		  'scipy == 0.14.0',
		  'scikit-learn == 0.15.2'
	],
    package_dir={'': '..'},
    package_data={'': ['train/*/*',"word_freq.db"]},
    )
