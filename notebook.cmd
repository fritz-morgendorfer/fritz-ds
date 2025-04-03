@echo off

title jupyter @ fritz-ds

call activate fritz-ds
call conda env list

setlocal
set PROJECT_ROOT=%cd%

call jupyter notebook
