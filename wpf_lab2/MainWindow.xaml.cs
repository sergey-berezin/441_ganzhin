using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using Ookii.Dialogs.Wpf;
using System.Threading;
using ArcFace;
using ImageControl = System.Windows.Controls.Image;
using System.IO;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp;
using Image = SixLabors.ImageSharp.Image;
using SixLabors.ImageSharp.PixelFormats;


namespace Lab2vb1
{
    public partial class MainWindow : Window
    {
        private string FFolderPath = "";
        private string SFolderPath = "";
        public MainWindow()
        {
            InitializeComponent();
            CancellationButton.IsEnabled = false;   
        }
        private readonly NNForSimilarity arcface = new();

        private CancellationTokenSource cts;

        private VistaFolderBrowserDialog folderDialog = new();

        private void FolderButton_Click(object sender, RoutedEventArgs e)
        {
            Button source = (Button)e.Source;
            folderDialog.ShowDialog();
            var FolderPath = folderDialog.SelectedPath;

            if (source.Name == "FirstFolderButton")
            {
                FFolderPath = FolderPath;
                GetImages(FolderPath, FirstList);
            }
            else if (source.Name == "SecondFolderButton")
            {
                SFolderPath = FolderPath;
                GetImages(FolderPath, SecondList);
            }

        }
        private void GetImages(string path, ListBox list)
        {
            list.Items.Clear();
            var imagePaths = Directory.GetFiles(path).Where(path => path.EndsWith(".png"));

            foreach (var imgpath in imagePaths)
            {
                StackPanel panel = new StackPanel();
                panel.Orientation = Orientation.Horizontal;

                ImageControl image = new()
                {
                    Source = new BitmapImage(new Uri(imgpath)),
                    Width = 80
                };
                panel.Children.Add(image);
                TextBlock textBlock = new TextBlock();

                textBlock.Text = System.IO.Path.GetFileName(imgpath);
                panel.Children.Add(textBlock);

                list.Items.Add(panel);
            }
        }

        public async void Calculate(object sender, RoutedEventArgs e)
        {
            CalculationProgressBar.Value = 0;
            CalculateButton.IsEnabled = false;
            CancellationButton.IsEnabled = true;
            cts = new CancellationTokenSource();

            var firstSelectedItem = (StackPanel)FirstList.SelectedItem;
            var secondSelectedItem = (StackPanel)SecondList.SelectedItem;
            CalculationProgressBar.Value += 10;


            using var face1 = Image.Load<Rgb24>(FFolderPath + "\\" + firstSelectedItem.Children.OfType<TextBlock>().Last().Text);
            face1.Mutate(x => x.Resize(112, 112));
            CalculationProgressBar.Value += 10;

            using var face2 = Image.Load<Rgb24>(SFolderPath + "\\" + secondSelectedItem.Children.OfType<TextBlock>().Last().Text);
            face2.Mutate(x => x.Resize(112, 112));
            CalculationProgressBar.Value += 10;



            var results = await arcface.GetDistanceNSimilarityAsync(face1, face2, cts.Token);

            CalculationProgressBar.Value += 50;

            if (!cts.Token.IsCancellationRequested)
            {
                var ListRes = results.ToList();
                SimilarityBlock.Text = ListRes[1].Item2.ToString();
                DistanceBlock.Text = ListRes[0].Item2.ToString();
                CalculationProgressBar.Value += 20;
            }
            else
            {
                CancellationRequested();
            } 
            CalculateButton.IsEnabled = true;
            CancellationButton.IsEnabled = false;
        }

        public void Cancellation(object sender, RoutedEventArgs e)
        {
            if(cts != null)
            {
                cts.Cancel();
            }
            CancellationRequested();
        }
        public void CancellationRequested()
        {
            CalculationProgressBar.Value = 0;
            SimilarityBlock.Text = "null";
            DistanceBlock.Text = "null";
        }

        private void List_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (FirstList.SelectedItems.Count != 0 && SecondList.SelectedItems.Count != 0)
            {
                CalculateButton.IsEnabled = true;
            }
        }
    }

}
