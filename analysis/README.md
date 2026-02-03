# GstPEAQ Install

Install all required packages:

```bash
sudo pacman -S \
  base-devel \
  gstreamer \
  gst-plugins-base \
  gst-plugins-good \
  python-gobject \
  fftw \
  gtk-doc

git clone https://github.com/HSU-ANT/gstpeaq.git
cd gstpeaq

touch ChangeLog

autoreconf -fi

./configure --prefix=/usr \
  --disable-man \
  --disable-gtk-doc

make
sudo make install
```

Verify GStreamer plugin install and python env:

```bash
gst-inspect-1.0 peaq

python -c "import gi; print('OK')"
```

