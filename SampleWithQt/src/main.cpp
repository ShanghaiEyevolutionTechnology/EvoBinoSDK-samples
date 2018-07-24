#include "mainwindow.h"
#include <QApplication>

#ifdef EVO_ENV_WINDOWS
#include <windows.h>
#include <stdio.h>
#endif

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

#ifdef EVO_ENV_WINDOWS
    // create a separate new console window
    AllocConsole();

    // attach the new console to this application's process
    AttachConsole(GetCurrentProcessId());

    // reopen the std I/O streams to redirect I/O to the new console
    freopen("CON", "w", stdout);
    freopen("CON", "w", stderr);
    freopen("CON", "r", stdin);
#endif

    MainWindow w;
    w.show();

    return a.exec();
}
