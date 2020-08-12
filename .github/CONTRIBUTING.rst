============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every
little bit helps, and credit will always be given.

You can contribute in many ways:

Report Bugs
===========

Report bugs at https://github.com/opencobra/optlang/issues.

If you are reporting a bug, please follow the template guidelines. The more 
detailed your report, the easier and thus faster we can help you.

Fix Bugs
========

Look through the GitHub issues for bugs. Anything tagged with "bug"
and "help wanted" is open to whoever wants to implement it.

Implement Features
==================

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
===================

optlang could always use more documentation, whether as part of the
official documentation, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
===============

The best way to send feedback is to file an issue at
https://github.com/opencobra/optlang/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
============

Ready to contribute? Here's how to set up optlang for
local development.

1. Fork the https://github.com/opencobra/optlang
   repository on GitHub.
2. Clone your fork locally

   .. code-block:: console
   
       git clone git@github.com:your_name_here/optlang.git

3. Install your local copy into a a Python virtual environment.
   You can `read this guide to learn more
   <https://realpython.com/python-virtual-environments-a-primer/>`_
   about them and how to create one. Alternatively, particularly if you are a 
   Windows or Mac user, you can also use
   `Anaconda <https://docs.anaconda.com/anaconda/>`_. Assuming you have 
   virtualenvwrapper installed, this is how you set up your fork for local development

   .. code-block:: console
   
       mkvirtualenv my-env
       cd optlang/
       pip install -e .[development]

4. Create a branch for local development using the ``devel`` branch as a 
   starting point. Use ``fix`` or ``feat`` as a prefix

   .. code-block:: console
   
       git checkout devel
       git checkout -b fix-name-of-your-bugfix

   Now you can make your changes locally.

5. When you're done making changes, apply the quality assurance tools and check 
   that your changes pass our test suite. This is all included with tox

   .. code-block:: console
   
       make qa
       tox

6. Commit your changes and push your branch to GitHub. Please use `semantic
   commit messages <http://karma-runner.github.io/2.0/dev/git-commit-msg.html>`_.

   .. code-block:: console
   
       git add .
       git commit -m "fix: Your summary of changes"
       git push origin fix-name-of-your-bugfix

7. Open the link displayed in the message when pushing your new branch 
   in order to submit a pull request.

Pull Request Guidelines
=======================

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring.
