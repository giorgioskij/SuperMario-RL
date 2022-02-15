%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% sapthesis.cls                                                   %%
%% version 5.0                                                     %%
%% date: see below                                                 %%
%% author: Francesco Biccari                                       %%
%% contact: biccari@gmail.com                                      %%
%%                                                                 %%
%% This file is the LaTeX class for the thesis style               %%
%% of the "Sapienza University of Rome"                            %%
%%                                                                 %%
%% see README for license information and installation             %%
%% see sapthesis-doc.pdf for usage                                 %%
%% see the examples folder for document examples                   %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Identification %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\NeedsTeXFormat{LaTeX2e}[2009/01/01]
\ProvidesClass{sapthesis}%
[2022/01/01 Sapienza University of Rome thesis class]


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Options %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{xkeyval}
\newlength{\SAP@binding}
\newif\ifSAP@nodefaultfont \SAP@nodefaultfontfalse
\newif\ifSAP@romandiff \SAP@romandifffalse
\newif\ifSAP@customlanguage \SAP@customlanguagefalse
\newif\ifSAP@noexaminfo \SAP@noexaminfofalse
\newif\ifSAP@femaleauthor \SAP@femaleauthorfalse
\def\SAP@paper{a4paper}
\def\SAP@layout{a4paper}
\def\SAP@fontsize{11pt}

\newcommand{\SAP@customlanguage}{}
\define@key[sap]{}{italian}[]{\SAP@customlanguagetrue
  \renewcommand{\SAP@customlanguage}{\SAP@italian}}
\define@key[sap]{}{english}[]{\SAP@customlanguagetrue
  \renewcommand{\SAP@customlanguage}{\SAP@english}}
\define@key[sap]{}{noexaminfo}[]{\SAP@noexaminfotrue}
\define@key[sap]{}{fem}[]{\SAP@femaleauthortrue}
\define@key[sap]{}{nodefaultfont}[]{\SAP@nodefaultfonttrue}
\define@key[sap]{}{romandiff}[]{\SAP@romandifftrue}
\define@key[sap]{}{binding}[0pt]{\setlength{\SAP@binding}{#1}}
\define@key[sap]{}{draft}[]{%
\PassOptionsToClass{draft}{book}%
%\PassOptionsToPackage{showframe}{geometry}
}
\define@key[sap]{}{oneside}[]{\PassOptionsToClass{oneside}{book}}
\define@key[sap]{}{twoside}[]{}
\define@key[sap]{}{a4paper}[]{\def\SAP@paper{a4paper}}
\define@key[sap]{}{a5paper}[]{%
\def\SAP@paper{a5paper}\def\SAP@layout{a5paper}\def\SAP@fontsize{10pt}}
\define@key[sap]{}{b5paper}[]{%
\def\SAP@paper{b5paper}\def\SAP@layout{b5paper}\def\SAP@fontsize{10pt}}
\define@choicekey[sap]{}{layout}[]{a4paper,a5paper,b5paper}[]{%
\def\SAP@layout{#1}\def\SAP@fontsize{10pt}}
\DeclareOptionX*{\ClassError{sapthesis}{Option '\CurrentOption'%
  \MessageBreak is not valid}{Remove '\CurrentOption' %
  from class options}}
\ProcessOptionsX[sap]<>


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Page layout %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\LoadClass[\SAP@paper,\SAP@fontsize]{book}
\RequirePackage[\SAP@paper]{geometry}

% A4
\def\SAP@layoutname{a4paper}
\ifx\SAP@layout\SAP@layoutname %
\geometry{textwidth=396pt,
  textheight=641pt,
  marginparsep=12pt,
  bindingoffset=\SAP@binding,
  vmarginratio=55:89,
  includehead,
  marginparwidth=72pt,
  heightrounded}
\if@twoside\geometry{hmarginratio=55:89}\fi
\fi
  
% A5
\def\SAP@layoutname{a5paper}
\ifx\SAP@layout\SAP@layoutname %
\geometry{textwidth=300pt,
  textheight=485pt,
  marginparsep=12pt,
  bindingoffset=\SAP@binding,
  vmarginratio=55:89,
  includehead,
  marginparwidth=42pt,
  heightrounded}
\if@twoside\geometry{hmarginratio=55:89}\fi
\def\SAP@papername{a4paper}
\ifx\SAP@paper\SAP@papername%
\geometry{layout=\SAP@layout,layouthoffset=2cm,layoutvoffset=2cm,showcrop}\fi
\fi

% B5
\def\SAP@layoutname{b5paper}
\ifx\SAP@layout\SAP@layoutname %
\geometry{textwidth=340pt,
  textheight=550pt,
  marginparsep=12pt,
  bindingoffset=\SAP@binding,
  vmarginratio=55:89,
  includehead,
  marginparwidth=42pt,
  heightrounded}
\if@twoside\geometry{hmarginratio=55:89}\fi
\def\SAP@papername{a4paper}
\ifx\SAP@paper\SAP@papername%
\geometry{layout=\SAP@layout,layouthoffset=2cm,layoutvoffset=2cm,showcrop}\fi
\fi

  



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Some settings%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\RequirePackage{ifxetex}
\ifxetex
  \RequirePackage{fontspec}
\else
  \ifSAP@nodefaultfont\relax\else
    \RequirePackage[T1]{fontenc}
    \RequirePackage{textcomp}
    \RequirePackage{lmodern}
  \fi
\fi
\RequirePackage{caption}
\captionsetup{indention=\parindent,labelfont={small,bf},labelsep=period,%
  tableposition=top,textfont={small}}
\RequirePackage{graphicx}
\RequirePackage{color}
\RequirePackage{booktabs}
\RequirePackage{amsmath}
\RequirePackage{etoolbox}



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Fancy headers %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% this two lines remove the headers in the empty pages
%% (instead of loading the emptypage package)
\def\cleardoublepage{\clearpage\if@twoside \ifodd\c@page\else
\thispagestyle{empty}\hbox{}\newpage\fi\fi}

\RequirePackage{fancyhdr}

%% definition of the main page style
\fancypagestyle{SAP@mainstyle}{%
\pagestyle{fancy}
\renewcommand{\chaptermark}[1]{\markboth{\ifnum \c@chapter>0 \thechapter.\ ##1 \else ##1 \fi}{}}
\renewcommand{\sectionmark}[1]{\markright{\thesection\ ##1}}
\fancyhf{}
\fancyhead[OR]{\small\bfseries\thepage}
\fancyhead[OL]{\small\bfseries\nouppercase{\rightmark}}
% This check is useful to avoid Warnings with oneside option
\if@twoside
\fancyhead[ER]{\small\bfseries\nouppercase{\leftmark}}
\fancyhead[EL]{\small\bfseries\thepage}
\fi
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0pt}}

%% re-definition of the plain style (first page of the chapters)
\fancypagestyle{plain}{%
\renewcommand{\headrulewidth}{0pt}
\fancyhf{}
%\fancyfoot[C]{\small\scshape\thepage}
\fancyhead[OR]{\small\bfseries\thepage}
% This check is useful to avoid Warnings with oneside option
\if@twoside
\fancyhead[EL]{\small\bfseries\thepage}
\fi
}

\g@addto@macro\frontmatter{\pagestyle{SAP@mainstyle}}
\g@addto@macro\mainmatter{\pagestyle{SAP@mainstyle}}

% fancy marginnote
% redefine the marginpar command
\def \@marginparreset {%
        \reset@font
        \small
        \itshape
        \@setminipage
}
\long\def\@xmpar[#1]#2{%
  \@savemarbox\@marbox{\raggedleft #1}%
  \@savemarbox\@currbox{\raggedright #2}%
  \@xympar}
\long\def\@ympar#1{%
  \@savemarbox\@marbox{\raggedleft #1}%
  \@savemarbox\@currbox{\raggedright #1}%
  \@xympar}



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% numbering objects per chapters (e.g. 1.1, 1.2, ...) %%%%%%%%%%%%%%

\numberwithin{equation}{chapter}
\numberwithin{figure}{chapter}
\numberwithin{table}{chapter}



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% redefinition of the \maketitle command to generate the titlepage %

%% GENERAL COMMANDS

% logo
\newif\ifSAP@sapienzalogo \SAP@sapienzalogofalse
\IfFileExists{sapienzalogo.pdf}{\SAP@sapienzalogotrue}{%
\ClassWarningNoLine{sapthesis}{Logo not found! You have to download it
manually. See the documentation or just the frontispiece.}
\RequirePackage{hyperref}}



\def\SAP@subtitle{}
\def\SAP@alttitle{}
\def\SAP@courseorganizerLabel{}
\def\SAP@courseLabel{}

\newcommand{\subtitle}[1]{\def\SAP@subtitle{#1}}
\newcommand{\alttitle}[1]{\def\SAP@alttitle{#1}}
\newcommand{\courseorganizer}[1]{\def\SAP@courseorganizerLabel{#1}}
\newcommand{\course}[1]{\def\SAP@courseLabel{#1}}

\def\SAP@copyyear{}
\newcommand{\copyyear}[1]{\def\SAP@copyyear{#1}}

\def\SAP@cycle{}
\newcommand{\cycle}[1]{\def\SAP@cycle{#1}}

\def\SAP@submitdate{}
\newcommand{\submitdate}[1]{\def\SAP@AcademicYear{#1}}
\let\AcademicYear\submitdate

\def\SAP@IDnumber{}
\newcommand{\IDnumber}[1]{\def\SAP@IDnumber{#1}}

%% Advisors
\newcount\SAP@advisorcount
\newtoks\SAP@advisortoks
\newcommand{\advisor}[1]{% 
  \ifnum\SAP@advisorcount=\z@ 
    \SAP@advisortoks={#1}% 
  \else 
    \SAP@advisortoks=\expandafter{\the\SAP@advisortoks\\[1mm]#1}% 
  \fi 
  \advance\SAP@advisorcount\@ne}

\newcommand{\customadvisorlabel}[1]{%
    \newcommand{\SAP@ThesisAdvisorLabel}{#1}
    \newcommand{\SAP@ThesisAdvisorsLabel}{#1}}
    
%% Co-Advisors or External Advisor 
\newcount\SAP@coadvisorcount
\newtoks\SAP@coadvisortoks
\newcommand{\coadvisor}[1]{%
  \ifnum\SAP@coadvisorcount=\z@ 
    \SAP@coadvisortoks={#1}% 
  \else 
    \SAP@coadvisortoks=\expandafter{\the\SAP@coadvisortoks\\[1mm]#1}% 
  \fi 
  \advance\SAP@coadvisorcount\@ne}

\newcommand{\customcoadvisorlabel}[1]{%
    \newcommand{\SAP@ThesisCoAdvisorLabel}{#1}
    \newcommand{\SAP@ThesisCoAdvisorsLabel}{#1}}

%% Reviewers
\def\SAP@ReviewerLabel{}
\newcommand{\reviewerlabel}[1]{\def\SAP@ReviewerLabel{#1}}
 
\newcount\SAP@reviewercount
\newtoks\SAP@reviewertoks
\newcommand{\reviewer}[1]{%
  \ifnum\SAP@reviewercount=\z@ 
    \SAP@reviewertoks={#1}% 
  \else 
    \SAP@reviewertoks=\expandafter{\the\SAP@reviewertoks\\[1mm]#1}% 
  \fi 
  \advance\SAP@reviewercount\@ne}  
  
  
  
%% Directors
\newcount\SAP@directorcount
\newtoks\SAP@directortoks
\newcommand{\director}[1]{% 
  \ifnum\SAP@directorcount=\z@ 
    \SAP@directortoks={#1}% 
  \else 
    \SAP@directortoks=\expandafter{\the\SAP@directortoks\\[0.75mm]#1}% 
  \fi 
  \advance\SAP@directorcount\@ne}

\newcommand{\customdirectorlabel}[1]{%
    \newcommand{\SAP@ThesisDirectorLabel}{#1}
    \newcommand{\SAP@ThesisDirectorsLabel}{#1}}

%% Examiners
\newcount\SAP@examinercount
\newtoks\SAP@examinertoks
\newcommand{\examiner}[2][]{% 
  \ifnum\SAP@examinercount=\z@ 
    \SAP@examinertoks={#2 (\SAP@chairman\if|#1|\relax\else, #1\fi)}% 
  \else 
    \SAP@examinertoks=\expandafter{\the\SAP@examinertoks\\[0.75mm]#2 \if|#1|\relax\else (#1)\fi}% 
  \fi 
  \advance\SAP@examinercount\@ne}

\def\SAP@examdate{}
\newcommand{\examdate}[1]{\def\SAP@examdate{#1}}

\def\SAP@thesistype{}
\newcommand{\thesistype}[1]{\def\SAP@thesistype{#1}}

\def\SAP@versiondate{}
\newcommand{\versiondate}[1]{\def\SAP@versiondate{#1}}

\def\SAP@authoremail{}
\newcommand{\authoremail}[1]{\def\SAP@authoremail{#1}}

\def\SAP@website{}
\newcommand{\website}[1]{\def\SAP@website{#1}}

\def\SAP@ISBN{}
\newcommand{\ISBN}[1]{\def\SAP@ISBN{#1}}

\def\SAP@copyrightstatement{}
\newcommand{\copyrightstatement}[1]{\def\SAP@copyrightstatement{#1}}

\def\SAP@extrainfo{}
\newcommand{\extrainfo}[1]{\def\SAP@extrainfo{#1}}



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% maketitle commands for A4 (A), A5 (B), B5 (C) formats 

\newcommand{\SAP@maketitleA}{

\begin{titlepage}

\linespread{1}\selectfont
\newlength{\SAP@parboxindent}
\newlength{\SAP@parboxwidth}
\setlength{\SAP@parboxindent}{1.72cm}
\setlength{\SAP@parboxwidth}{\dimexpr\textwidth-\SAP@parboxindent\relax}
\newcommand{\SAP@parbox}[2][]{\hspace{\SAP@parboxindent}%
\parbox[t][##1][t]{\SAP@parboxwidth}{##2}}
\parindent=0pt
\sffamily

\vspace*{-10mm}

\ifSAP@sapienzalogo
	\includegraphics[width=5cm]{sapienzalogo}
\else
	\parbox[b][3cm][c]{0.49\linewidth}{\hspace{-1cm}
		\setlength{\unitlength}{1mm}
		\begin{picture}(65,27.3)(0,0)
		\small
		\put(0,0){\framebox(65,27.3)[0,0]{}}
		\put(2,23){\makebox(0,0)[l]{WARNING! ATTENZIONE!}}
		\put(2,18){\makebox(0,0)[l]{Sapienza logo not found!}}
		\put(2,11){\href{http://biccari.altervista.org/c/informatica/latex/sapienzalogo.pdf}{Click here to download the Sapienza logo}}
		\put(2,5.5){Then copy sapienzalogo.pdf in the folder}
		\put(2,1.5){of your main .tex file.}
		\end{picture}
	}
\fi

\nointerlineskip

%\nointerlineskip
%\vspace{\stretch{1}}
%\nointerlineskip

%\fbox{%
\SAP@parbox[12.8cm]{
    \nointerlineskip
    \vspace*{7mm}
    \vspace{\stretch{1}}
    \nointerlineskip
    
    {\LARGE\linespread{1.1}\selectfont\textcolor{sapred}{\@title}\par}
    
    \if|\SAP@subtitle|
        \if|\SAP@alttitle|
            \relax
        \else
            \vspace{1cm}
            {\LARGE\linespread{1.1}\selectfont\textcolor{sapred}{\SAP@alttitle}\par}
        \fi
    \else
        \vspace{0.4cm}
        {\large\linespread{1.1}\selectfont\textcolor{sapred}{\SAP@subtitle}\par}
    \fi
    
    \nointerlineskip
    \vspace{\stretch{1}}
    \vspace*{7mm}
    \nointerlineskip}
%}

\nointerlineskip
%\vspace{\stretch{1}}

%\fbox{%
\SAP@parbox{%
\textcolor{sapred}{\SAP@courseorganizerLabel\\[1mm]
\SAP@courseLabel\ \if|\SAP@cycle|\relax\else (\SAP@cycle\ \SAP@cycleLabel)\fi}

\vspace*{9mm}

{\bfseries\@author}\\[1mm]
\SAP@IDnumberLabel\ \SAP@IDnumber

\vspace*{9mm}

\makebox[11.9cm][l]{%
  \parbox[t]{5.9cm}{%
    \ifcase\SAP@advisorcount
      \ClassError{sapthesis}{You must specify at least one advisor in the preamble}%
      {E.g. \protect\advisor{Prof. Name Surname}}
    \or
      \SAP@ThesisAdvisorLabel\\[1mm]
    \else
      \SAP@ThesisAdvisorsLabel\\[1mm]
    \fi
    \the\SAP@advisortoks}
    \hspace{15mm}
    \parbox[t]{5.9cm}{%
    \ifcase\SAP@coadvisorcount
      \relax
    \or
      \SAP@ThesisCoAdvisorLabel\\[1mm]
    \else
      \SAP@ThesisCoAdvisorsLabel\\[1mm]
    \fi
    \the\SAP@coadvisortoks}
}%

\ifcase\SAP@directorcount
    \relax
\or
    \vspace*{9mm}
    \SAP@ThesisDirectorLabel\\[1mm]
    \the\SAP@directortoks
\else
    \vspace*{9mm}
    \SAP@ThesisDirectorsLabel\\[1mm]
    \the\SAP@directortoks
\fi

}
%}

\nointerlineskip

\vfill


%\fbox{%
\SAP@parbox{%
%\ifnum\SAP@directorcount=\z@
%    \vspace*{20mm}
%\else
%    \vspace*{10mm}
%\fi
\SAP@AcademicYearLabel\ \SAP@AcademicYear
}
%}
\end{titlepage}
}


\newcommand{\SAP@maketitleB}{
\begin{titlepage}

\linespread{1}\selectfont
\newlength{\SAP@parboxindent}
\newlength{\SAP@parboxwidth}
\setlength{\SAP@parboxindent}{1.38cm}
\setlength{\SAP@parboxwidth}{\dimexpr\textwidth-\SAP@parboxindent\relax}
\newcommand{\SAP@parbox}[2][]{\hspace{\SAP@parboxindent}%
\parbox[t][##1][t]{\SAP@parboxwidth}{##2}}
\parindent=0pt
\sffamily

\vspace*{-10mm}

\ifSAP@sapienzalogo
	\includegraphics[width=4cm]{sapienzalogo}
\else
	\parbox[b][3cm][c]{0.49\linewidth}{\hspace{-1cm}
		\setlength{\unitlength}{1mm}
		\begin{picture}(65,27.3)(0,0)
		\small
		\put(0,0){\framebox(65,27.3)[0,0]{}}
		\put(2,23){\makebox(0,0)[l]{WARNING! ATTENZIONE!}}
		\put(2,18){\makebox(0,0)[l]{Sapienza logo not found!}}
		\put(2,11){\href{http://biccari.altervista.org/c/informatica/latex/sapienzalogo.pdf}{Click here to download the Sapienza logo}}
		\put(2,5.5){Then copy sapienzalogo.pdf in the folder}
		\put(2,1.5){of your main .tex file.}
		\end{picture}
	}
\fi

\nointerlineskip

%\nointerlineskip
%\vspace{\stretch{1}}
%\nointerlineskip

%\fbox{%
\SAP@parbox[9.3cm]{
    \nointerlineskip
    \vspace*{7mm}
    \vspace{\stretch{1}}
    \nointerlineskip
    
    {\Large\linespread{1.1}\selectfont\textcolor{sapred}{\@title}\par}
    
    \if|\SAP@subtitle|
        \if|\SAP@alttitle|
            \relax
        \else
            \vspace{7mm}
            {\Large\linespread{1.1}\selectfont\textcolor{sapred}{\SAP@alttitle}\par}
        \fi
    \else
        \vspace{0.4cm}
        {\normalsize\linespread{1.1}\selectfont\textcolor{sapred}{\SAP@subtitle}\par}
    \fi
    
    \nointerlineskip
    \vspace{\stretch{1}}
    \vspace*{7mm}
    \nointerlineskip}
%}

\nointerlineskip
%\vspace{\stretch{1}}

%\fbox{%
\SAP@parbox{%
\textcolor{sapred}{\SAP@courseorganizerLabel\\[1mm]
\SAP@courseLabel\ \if|\SAP@cycle|\relax\else (\SAP@cycle\ \SAP@cycleLabel)\fi}

\vspace*{7mm}

{\bfseries\@author}\\[1mm]
\SAP@IDnumberLabel\ \SAP@IDnumber

\vspace*{7mm}

\raggedright
\makebox[0cm][l]{%
  \parbox[t]{4cm}{%
    \ifcase\SAP@advisorcount
      \ClassError{sapthesis}{You must specify at least one advisor in the preamble}%
      {E.g. \protect\advisor{Prof. Name Surname}}
    \or
      \SAP@ThesisAdvisorLabel\\[1mm]
    \else
      \SAP@ThesisAdvisorsLabel\\[1mm]
    \fi
    \the\SAP@advisortoks}
    \hspace{2mm}
    \parbox[t]{4cm}{%
    \ifcase\SAP@coadvisorcount
      \relax
    \or
      \SAP@ThesisCoAdvisorLabel\\[1mm]
    \else
      \SAP@ThesisCoAdvisorsLabel\\[1mm]
    \fi
    \the\SAP@coadvisortoks}
}%

\ifcase\SAP@directorcount
    \relax
\or
    \vspace*{7mm}
    \SAP@ThesisDirectorLabel\\[1mm]
    \the\SAP@directortoks
\else
    \vspace*{7mm}
    \SAP@ThesisDirectorsLabel\\[1mm]
    \the\SAP@directortoks
\fi

}
%}

\nointerlineskip

\vfill


%\fbox{%
\SAP@parbox{%
%\ifnum\SAP@directorcount=\z@
%    \vspace*{20mm}
%\else
%    \vspace*{10mm}
%\fi
\SAP@AcademicYearLabel\ \SAP@AcademicYear
}
%}
\end{titlepage}
}


\newcommand{\SAP@maketitleC}{
\begin{titlepage}

\linespread{1}\selectfont
\newlength{\SAP@parboxindent}
\newlength{\SAP@parboxwidth}
\setlength{\SAP@parboxindent}{1.38cm}
\setlength{\SAP@parboxwidth}{\dimexpr\textwidth-\SAP@parboxindent\relax}
\newcommand{\SAP@parbox}[2][]{\hspace{\SAP@parboxindent}%
\parbox[t][##1][t]{\SAP@parboxwidth}{##2}}
\parindent=0pt
\sffamily

\vspace*{-10mm}

\ifSAP@sapienzalogo
	\includegraphics[width=4cm]{sapienzalogo}
\else
	\parbox[b][3cm][c]{0.49\linewidth}{\hspace{-1cm}
		\setlength{\unitlength}{1mm}
		\begin{picture}(65,27.3)(0,0)
		\small
		\put(0,0){\framebox(65,27.3)[0,0]{}}
		\put(2,23){\makebox(0,0)[l]{WARNING! ATTENZIONE!}}
		\put(2,18){\makebox(0,0)[l]{Sapienza logo not found!}}
		\put(2,11){\href{http://biccari.altervista.org/c/informatica/latex/sapienzalogo.pdf}{Click here to download the Sapienza logo}}
		\put(2,5.5){Then copy sapienzalogo.pdf in the folder}
		\put(2,1.5){of your main .tex file.}
		\end{picture}
	}
\fi

\nointerlineskip

%\nointerlineskip
%\vspace{\stretch{1}}
%\nointerlineskip

%\fbox{%
\SAP@parbox[11.4cm]{
    \nointerlineskip
    \vspace*{7mm}
    \vspace{\stretch{1}}
    \nointerlineskip
    
    {\Large\linespread{1.1}\selectfont\textcolor{sapred}{\@title}\par}
    
    \if|\SAP@subtitle|
        \if|\SAP@alttitle|
            \relax
        \else
            \vspace{1cm}
            {\Large\linespread{1.1}\selectfont\textcolor{sapred}{\SAP@alttitle}\par}
        \fi
    \else
        \vspace{0.4cm}
        {\normalsize\linespread{1.1}\selectfont\textcolor{sapred}{\SAP@subtitle}\par}
    \fi
    
    \nointerlineskip
    \vspace{\stretch{1}}
    \vspace*{7mm}
    \nointerlineskip}
%}

\nointerlineskip
%\vspace{\stretch{1}}

%\fbox{%
\SAP@parbox{%
\textcolor{sapred}{\SAP@courseorganizerLabel\\[1mm]
\SAP@courseLabel\ \if|\SAP@cycle|\relax\else (\SAP@cycle\ \SAP@cycleLabel)\fi}

\vspace*{7mm}

{\bfseries\@author}\\[1mm]
\SAP@IDnumberLabel\ \SAP@IDnumber

\vspace*{7mm}

\raggedright
\makebox[0cm][l]{%
  \parbox[t]{5cm}{%
    \ifcase\SAP@advisorcount
      \ClassError{sapthesis}{You must specify at least one advisor in the preamble}%
      {E.g. \protect\advisor{Prof. Name Surname}}
    \or
      \SAP@ThesisAdvisorLabel\\[1mm]
    \else
      \SAP@ThesisAdvisorsLabel\\[1mm]
    \fi
    \the\SAP@advisortoks}
    \hspace{2mm}
    \parbox[t]{5cm}{%
    \ifcase\SAP@coadvisorcount
      \relax
    \or
      \SAP@ThesisCoAdvisorLabel\\[1mm]
    \else
      \SAP@ThesisCoAdvisorsLabel\\[1mm]
    \fi
    \the\SAP@coadvisortoks}
}%

\ifcase\SAP@directorcount
    \relax
\or
    \vspace*{7mm}
    \SAP@ThesisDirectorLabel\\[1mm]
    \the\SAP@directortoks
\else
    \vspace*{7mm}
    \SAP@ThesisDirectorsLabel\\[1mm]
    \the\SAP@directortoks
\fi

}
%}

\nointerlineskip

\vfill


%\fbox{%
\SAP@parbox{%
%\ifnum\SAP@directorcount=\z@
%    \vspace*{20mm}
%\else
%    \vspace*{10mm}
%\fi
\SAP@AcademicYearLabel\ \SAP@AcademicYear
}
%}
\end{titlepage}
}



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Macro for the back of the title page %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\def\SAP@composebacktitlepage{%

{\thispagestyle{empty}
\linespread{1}\selectfont
\setlength{\parindent}{0pt}

\null
\vspace*{1cm}
\if|\SAP@extrainfo|\relax\else
\SAP@extrainfo
\vspace*{1.5cm}
\fi

\ifSAP@noexaminfo\relax
\else
\if?\SAP@examdate?
  \ifcase\SAP@examinercount
    \SAP@ThesisNotDefensedLabel%
  \else
    \ClassError{sapthesis}{You have specified one or more examiners but not
    the date of the final exam}{E.g. \protect\examdate{17 July 2015}}
  \fi
\else
  \ifcase\SAP@examinercount
    \ClassError{sapthesis}{You have specified the final exam date but no examiner}{E.g. \protect\examiner{Prof. Giulio Cesare}}
  \else
    \SAP@ThesisDefensedLabelA\ \SAP@examdate\\
    \SAP@ThesisDefensedLabelB:\\[2mm]
    \the\SAP@examinertoks%
  \fi
\fi
\fi

\vspace{1cm}

\ifcase\SAP@reviewercount
  \relax
\else
  \SAP@ReviewerLabel:\\[2mm]
  \the\SAP@reviewertoks%
\fi

\vfill

\hrulefill
\vspace{2mm}

\small
\textbf{\def\\{}\def\newline{}\@title} \\[0.75mm]
\if|\SAP@thesistype|\relax\else\SAP@thesistype.\fi\ \SAP@sapienzaname \\[0.75mm]
\if|\SAP@ISBN|\relax\else ISBN: \SAP@ISBN\fi\\[0.75mm]
\if|\SAP@copyrightstatement|\copyright\ \SAP@copyyear\ \@author. \SAP@allRightsReservedLabel\else\SAP@copyrightstatement\fi\\[1.5mm]

\SAP@sapthesisInformationLabel.\\

\if|\SAP@versiondate|
  \relax%
\else
  \SAP@versionLabel: \SAP@versiondate \\[2mm]%
\fi
\if|\SAP@website|
  \relax%
\else
  \SAP@websiteLabel: \@ifundefined{href}{\SAP@website}{\href{\SAP@website}{\SAP@website}} \\[2mm]%
\fi
\SAP@emailLabel: \@ifundefined{href}{\SAP@authoremail}{\href{mailto:\SAP@authoremail}{\SAP@authoremail}}\par

}
\cleardoublepage
}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Choosing the right \maketitle according to the thesis type %%%%%%%

\renewcommand{\maketitle}{%
% Errors:
\if|\SAP@courseorganizerLabel|\ClassError{sapthesis}{You must specify the course %
  organizer}{Example: \protect\courseorganizer{Scuola di dottorato di ...}}\fi
\if|\SAP@courseLabel|\ClassError{sapthesis}{You must specify the %
  course}{Example: \protect\course{Corso di Laurea Magistrale in Fisica}}\fi
\if|\SAP@AcademicYear|\ClassError{sapthesis}{You must specify the %
  thesis submission date}{Example: \protect\submitdate{August 2012} The date must be in the form Month(name) Year(number)}\fi
\if|\SAP@copyyear|\ClassError{sapthesis}{You must specify the %
  copyright year}{Example: \protect\copyyear{2012}}\fi
\if|\SAP@IDnumber|\ClassError{sapthesis}{You must specify your %
  ID number (matricola)}{Example: \protect\IDnumber{123456}}\fi
\if|\SAP@authoremail|\ClassError{sapthesis}{You must specify the email of the author}{%
  Example: \protect\authoremail{author@email.com}}\fi
%titlepage
\def\SAP@layoutname{a4paper}
\ifx\SAP@layout\SAP@layoutname
\SAP@maketitleA
\fi
\def\SAP@layoutname{a5paper}
\ifx\SAP@layout\SAP@layoutname
\SAP@maketitleB
\fi
\def\SAP@layoutname{b5paper}
\ifx\SAP@layout\SAP@layoutname
\SAP@maketitleC
\fi
\clearpage
\SAP@composebacktitlepage}



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% DEDICATION COMMAND %%
\providecommand*{\dedication}[1]{%
\cleardoublepage
\thispagestyle{empty}
\null\vspace{\stretch{0.26}}
\begin{flushright}
\textit{#1}
\end{flushright}
\vspace{\stretch{0.74}}\null}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ABSTRACT ENVIRONMENT %%
\newenvironment{abstract}[1][\SAP@AbstractLabel]
{\cleardoublepage\thispagestyle{plain}
\noindent{\Large\bfseries #1}\par\vspace{0.5cm}\noindent\ignorespaces}
{\null\ignorespacesafterend}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ACKNOWLEDGMENTS ENVIRONMENT %%
\newenvironment{acknowledgments}[1][\SAP@AcknowledgmentsLabel]
{\cleardoublepage\thispagestyle{plain}
\noindent{\Large\bfseries #1}\par\vspace{0.5cm}\itshape\noindent\ignorespaces}
{\null\ignorespacesafterend}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% LANGUAGE COMMANDS %%

\newcommand{\SAP@english}{%
  \newcommand{\SAP@CandidateLabel}{Candidate}
  \newcommand{\SAP@CandidateLabelFemale}{Candidate}
  \newcommand{\SAP@IDnumberLabel}{ID number}
  \newcommand{\SAP@ThesisAdvisorLabel}{Advisor}
  \newcommand{\SAP@ThesisAdvisorsLabel}{Advisors}
  \providecommand{\SAP@ThesisCoAdvisorLabel}{Co-Advisor}
  \providecommand{\SAP@ThesisCoAdvisorsLabel}{Co-Advisors}
  \providecommand{\SAP@ThesisDirectorLabel}{Director}
  \providecommand{\SAP@ThesisDirectorsLabel}{Directors}
  \newcommand{\SAP@cycleLabel}{cycle}
  \newcommand{\SAP@AcademicYearLabel}{Academic Year}
  \newcommand{\SAP@ThesisNotDefensedLabel}{Thesis not yet defended}
  \newcommand{\SAP@ThesisDefensedLabelA}{Thesis defended on}
  \newcommand{\SAP@ThesisDefensedLabelB}{in front of a Board of Examiners composed by}
  \newcommand{\SAP@chairman}{chairman}
  \newcommand{\SAP@sapienzaname}{Sapienza University of Rome}
  \newcommand{\SAP@versionLabel}{Version}
  \newcommand{\SAP@websiteLabel}{Website}
  \newcommand{\SAP@AcknowledgmentsLabel}{Acknowledgments}
  \newcommand{\SAP@AbstractLabel}{Abstract}
  \newcommand{\SAP@allRightsReservedLabel}{All rights reserved}
  \newcommand{\SAP@sapthesisInformationLabel}{This thesis has been typeset by \LaTeX\ and the Sapthesis class}
  \newcommand{\SAP@emailLabel}{Author's email}}

\newcommand{\SAP@italian}{%
  \newcommand{\SAP@CandidateLabel}{Candidato}
  \newcommand{\SAP@CandidateLabelFemale}{Candidata}
  \newcommand{\SAP@IDnumberLabel}{Matricola}
  \newcommand{\SAP@ThesisAdvisorLabel}{Relatore}
  \newcommand{\SAP@ThesisAdvisorsLabel}{Relatori}
  \providecommand{\SAP@ThesisCoAdvisorLabel}{Correlatore}
  \providecommand{\SAP@ThesisCoAdvisorsLabel}{Correlatori}
  \providecommand{\SAP@ThesisDirectorLabel}{Direttore}
  \providecommand{\SAP@ThesisDirectorsLabel}{Direttori}
  \newcommand{\SAP@cycleLabel}{ciclo}
  \newcommand{\SAP@AcademicYearLabel}{Anno Accademico}
  \newcommand{\SAP@ThesisNotDefensedLabel}{Tesi non ancora discussa}
  \newcommand{\SAP@ThesisDefensedLabelA}{Tesi discussa il}
  \newcommand{\SAP@ThesisDefensedLabelB}{di fronte a una commissione esaminatrice composta da}
  \newcommand{\SAP@chairman}{presidente}
  \newcommand{\SAP@sapienzaname}{Sapienza Universit\`a di Roma}
  \newcommand{\SAP@versionLabel}{Versione}
  \newcommand{\SAP@websiteLabel}{Sito web}
  \newcommand{\SAP@AcknowledgmentsLabel}{Ringraziamenti}
  \newcommand{\SAP@AbstractLabel}{Sommario}
  \newcommand{\SAP@allRightsReservedLabel}{Tutti i diritti riservati}
  \newcommand{\SAP@sapthesisInformationLabel}{Questa tesi \`e stata composta con \LaTeX\ e la classe Sapthesis}
  \newcommand{\SAP@emailLabel}{Email dell'autore}}

  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% USEFUL COMMANDS %%

\newcommand*{\SAP@upbox}[1]{\textrm{\upshape#1}} 
\AtBeginDocument{

  \definecolor{sapred}{rgb}{0.5098039,0.1411765,0.2}

  % scritte in funzione della lingua del documento. Default inglese.
  \ifSAP@customlanguage
    \SAP@customlanguage
  \else
    \@ifpackageloaded{babel}{\csname SAP@\languagename\endcsname}{%
      \@ifpackageloaded{polyglossia}{\csname SAP@\languagename\endcsname}{\SAP@english}}
  \fi

  % degree 
  \providecommand{\degree}{\SAP@upbox{\textdegree}} 
  \providecommand{\g}{\degree} 

  % gradi Celsius
  \providecommand{\celsius}{\ensuremath{\SAP@upbox{\textdegree}\kern-\scriptspace\mathrm{C}}}
  \ifxetex
    \ProvideTextCommandDefault{\C}{\celsius}
  \else
  
    \providecommand{\C}{\celsius}
  \fi
  
  % Angstrom 
  \providecommand{\angstrom}{\SAP@upbox{\AA}} 
  \providecommand{\A}{\angstrom} 
  
  % micro 
  \providecommand{\micro}{\SAP@upbox{\textmu}} 
  
  % Ohm 
  \providecommand{\ohm}{\SAP@upbox{\textohm}} 
  
  % times (only math) 
  \providecommand{\x}{\times} 
  
  % measurement unit 
  \providecommand*{\un}[1]{\ensuremath{\mathrm{\,#1}}} 

  % Napier's number 
  \providecommand{\eu}{\ensuremath{\mathrm{e}}} 
  
  % imaginary unit 
  \providecommand{\iu}{\ensuremath{\mathrm{i}}} 
  
  % roman subscript 
  \providecommand*{\rb}[1]{\ensuremath{_\mathrm{#1}}} 
  
  % roman superscript 
  \providecommand*{\rp}[1]{\ensuremath{^\mathrm{#1}}} 

  % differential (only math)
  \ifSAP@romandiff
    \providecommand{\di}{\mathop{}\!\mathrm{d}}
  \else
    \providecommand{\di}{\mathop{}\!d}
  \fi
  
  % derivative and partial derivative (only math)
  % see http://www.guit.sssup.it/phpbb/viewtopic.php?p=42199#42199
  \providecommand{\der}[2]{\begingroup 
    \@tempswafalse\toks@={}\count@=\z@ 
    \@for\next:=#2\do 
      {\expandafter\check@var\next\@nil 
       \advance\count@\der@exp 
       \if@tempswa 
         \toks@=\expandafter{\the\toks@\,}% 
       \else 
         \@tempswatrue 
       \fi 
       \toks@=\expandafter{\the\expandafter\toks@\expandafter\di\der@var}}%
    \frac{\di\ifnum\count@=\@ne\else^{\number\count@}\fi#1}{\the\toks@}% 
    \endgroup} 
  \def\check@var{\@ifstar{\mult@var}{\one@var}} 
  \def\mult@var#1#2\@nil{\def\der@var{#2^{#1}}\def\der@exp{#1}} 
  \def\one@var#1\@nil{\def\der@var{#1}\chardef\der@exp\@ne}
  
  \providecommand{\pder}[2]{\begingroup
    \@tempswafalse\toks@={}\count@=\z@
    \@for\next:=#2\do
      {\expandafter\check@var\next\@nil
       \advance\count@\der@exp
       \if@tempswa
         \toks@=\expandafter{\the\toks@\,}%
       \else
         \@tempswatrue
       \fi
       \toks@=\expandafter{\the\expandafter\toks@\expandafter\partial\der@var}}%
    \frac{\partial\ifnum\count@=\@ne\else^{\number\count@}\fi#1}{\the\toks@}%
    \endgroup}
  \def\check@var{\@ifstar{\mult@var}{\one@var}} 
  \def\mult@var#1#2\@nil{\def\der@var{#2^{#1}}\def\der@exp{#1}} 
  \def\one@var#1\@nil{\def\der@var{#1}\chardef\der@exp\@ne}

  % subscript and superscript in text mode
  \DeclareRobustCommand*\textsubscript[1]{\@textsubscript{\selectfont#1}} 
  \def\@textsubscript#1{{\m@th\ensuremath{_{\mbox{\fontsize\sf@size\z@#1}}}}}
  \providecommand*{\tb}[1]{\textsubscript{#1}}
  \providecommand*{\tp}[1]{\textsuperscript{#1}}
}