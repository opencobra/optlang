.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every
little bit helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
======================

Report Bugs
-----------

Report bugs at https://github.com/biosustain/optlang/issues.

If you are reporting a bug, please follow the presented issue template since 
it is designed to ultimately make helping you easier and thus faster.

Fix Bugs
--------

Look through the GitHub issues for bugs. Anything tagged with "bug"
and "help wanted" is open to whoever wants to implement it.

Implement Features
------------------

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
-------------------

As any open source project, optlang could always use more 
and better documentation, whether as part of the official docs, in docstrings, or even on the web in blog posts,
articles.

Submit Feedback
---------------

The best way to send feedback is to file an issue at https://github.com/biosustain/optlang/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome ;)

Get Started!
------------

Ready to contribute? Here's how to set up ``optlang`` 
for local development.

1. Fork the ``optlang`` repo on GitHub.
2. Clone your fork locally::

    git clone git@github.com:<your_name_here>/optlang.git

3. Install your local copy into a virtualenv. Assuming you have virtualenvwrapper installed, this is how you set up your fork for local development::

    mkvirtualenv optlang
    cd optlang/
    pip install -e .

4. Create a branch for local development::

    git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass the quality 
control::

    tox

To get tox, just pip install it into your virtualenv.

6. Commit your changes using `semantic commit messages <https://seesparkbox.com/foundry/semantic_commit_messages>`__ and push your branch to GitHub::

    git add .
    git commit -m "feat: your detailed description of your changes"
    git push origin name-of-your-bugfix-or-feature

7. Submit a pull request to this repository through the GitHub website.
